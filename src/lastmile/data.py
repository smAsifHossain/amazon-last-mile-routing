"""Routes of the challenge dataset.

The raw files are single JSON objects of up to 1.8 GB, far more than fits in
memory once parsed. They are therefore read one route at a time and converted
once into a compact file per route (``scripts/prepare_data.py``). Everything else
in the project works on those files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from . import paths

_WHITESPACE = re.compile(r"[ \t\r\n]*")


@dataclass
class Route:
    """One delivery route. Stops are numbered 0..n-1 and stop 0 is the station."""

    id: str
    station_code: str
    date: str
    departure_time_utc: str
    capacity_cm3: float
    quality: str | None  # "High" / "Medium" / "Low"; only in the training set
    stop_ids: list[str]
    lat: np.ndarray
    lng: np.ndarray
    zone_ids: list[str | None]  # None for the station and for some drop-offs
    times: np.ndarray  # travel time in seconds, [origin, destination]
    n_packages: np.ndarray
    service_time: np.ndarray  # planned seconds at the stop, all packages
    window_start: np.ndarray  # seconds after departure; NaN without a time window
    window_end: np.ndarray
    actual: list[int] | None = None  # driver's sequence of stop numbers, starts with 0

    @property
    def n_stops(self) -> int:
        return len(self.stop_ids)

    def to_ids(self, sequence: Sequence[int]) -> list[str]:
        return [self.stop_ids[i] for i in sequence]


# --------------------------------------------------------------------------- raw


def iter_json_object(path: Path, chunk_size: int = 1 << 22) -> Iterator[tuple[str, object]]:
    """Yield ``(key, value)`` for each entry of a top-level JSON object.

    Only one value is held in memory at a time, so the file can be much larger
    than memory. Values must be objects or arrays.
    """
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as handle:
        buffer = handle.read(chunk_size)
        pos = _WHITESPACE.match(buffer, 0).end()
        if buffer[pos : pos + 1] != "{":
            raise ValueError(f"{path}: expected a JSON object")
        pos += 1
        while True:
            # Find the next key, refilling the buffer when it runs out.
            while True:
                pos = _WHITESPACE.match(buffer, pos).end()
                if pos < len(buffer) and buffer[pos] == ",":
                    pos += 1
                    continue
                if pos < len(buffer):
                    break
                more = handle.read(chunk_size)
                if not more:
                    raise ValueError(f"{path}: unexpected end of file")
                buffer, pos = more, 0
            if buffer[pos] == "}":
                return
            # Decode "key": value; if the buffer ends mid-entry, read more and retry.
            while True:
                try:
                    key, end = decoder.raw_decode(buffer, pos)
                    end = _WHITESPACE.match(buffer, end).end()
                    if buffer[end] != ":":
                        raise ValueError(f"{path}: expected ':' after key {key!r}")
                    start = _WHITESPACE.match(buffer, end + 1).end()
                    value, end = decoder.raw_decode(buffer, start)
                    break
                except (json.JSONDecodeError, IndexError):
                    more = handle.read(chunk_size)
                    if not more:
                        raise ValueError(f"{path}: unexpected end of file") from None
                    buffer, pos = buffer[pos:] + more, 0
            yield key, value
            buffer, pos = buffer[end:], 0


def _seconds_after(departure: datetime, timestamp) -> float:
    if not isinstance(timestamp, str):
        return float("nan")
    return (datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") - departure).total_seconds()


def route_from_raw(route_id: str, raw: dict, times: dict, packages: dict | None, actual: dict | None) -> Route:
    """Build a route from its entries in the raw JSON files."""
    stops = raw["stops"]
    station = next(stop for stop, info in stops.items() if info["type"] == "Station")
    stop_ids = [station] + [stop for stop in stops if stop != station]
    n = len(stop_ids)
    departure = datetime.strptime(
        f"{raw['date_YYYY_MM_DD']} {raw['departure_time_utc']}", "%Y-%m-%d %H:%M:%S"
    )

    n_packages = np.zeros(n, dtype=np.int32)
    service_time = np.zeros(n)
    window_start = np.full(n, np.nan)
    window_end = np.full(n, np.nan)
    for i, stop in enumerate(stop_ids):
        for package in (packages or {}).get(stop, {}).values():
            n_packages[i] += 1
            planned = package.get("planned_service_time_seconds")
            if isinstance(planned, (int, float)) and planned == planned:
                service_time[i] += planned
            window = package.get("time_window", {})
            start = _seconds_after(departure, window.get("start_time_utc"))
            end = _seconds_after(departure, window.get("end_time_utc"))
            # Several packages at one stop: the stop must satisfy the tightest window.
            window_start[i] = np.fmax(window_start[i], start)
            window_end[i] = np.fmin(window_end[i], end)

    sequence = None
    if actual is not None:
        position = actual["actual"]
        number = {stop: i for i, stop in enumerate(stop_ids)}
        sequence = [number[stop] for stop in sorted(position, key=position.__getitem__)]

    return Route(
        id=route_id,
        station_code=raw["station_code"],
        date=raw["date_YYYY_MM_DD"],
        departure_time_utc=raw["departure_time_utc"],
        capacity_cm3=float(raw["executor_capacity_cm3"]),
        quality=raw.get("route_score"),
        stop_ids=stop_ids,
        lat=np.array([stops[stop]["lat"] for stop in stop_ids]),
        lng=np.array([stops[stop]["lng"] for stop in stop_ids]),
        zone_ids=[
            stops[stop]["zone_id"] if isinstance(stops[stop]["zone_id"], str) else None
            for stop in stop_ids
        ],
        times=np.array([[times[a][b] for b in stop_ids] for a in stop_ids], dtype=float),
        n_packages=n_packages,
        service_time=service_time,
        window_start=window_start,
        window_end=window_end,
        actual=sequence,
    )


def iter_raw_routes(
    routes_path: Path,
    travel_times_path: Path,
    packages_path: Path | None = None,
    actual_path: Path | None = None,
) -> Iterator[Route]:
    """Read routes one at a time from the raw JSON files."""
    actual = {}
    if actual_path:
        with open(actual_path, encoding="utf-8") as handle:
            actual = json.load(handle)
    package_entries = iter_json_object(packages_path) if packages_path else None
    times_entries = iter_json_object(travel_times_path)
    for route_id, raw in iter_json_object(routes_path):
        times_id, times = next(times_entries)
        packages = None
        if package_entries is not None:
            packages_id, packages = next(package_entries)
            if packages_id != route_id:
                raise ValueError(f"route order differs between files: {route_id} / {packages_id}")
        if times_id != route_id:
            raise ValueError(f"route order differs between files: {route_id} / {times_id}")
        yield route_from_raw(route_id, raw, times, packages, actual.get(route_id))


# --------------------------------------------------------------------- processed


def save_route(route: Route, folder: Path) -> Path:
    """Write one route as a compressed ``.npz`` file."""
    folder.mkdir(parents=True, exist_ok=True)
    # Travel times have one decimal, so tenths of a second are exact integers.
    tenths = np.rint(route.times * 10)
    if not np.allclose(tenths, route.times * 10, atol=1e-6):
        raise ValueError(f"{route.id}: travel times have more than one decimal")
    target = folder / f"{route.id}.npz"
    np.savez_compressed(
        target,
        meta=np.array(
            [route.id, route.station_code, route.date, route.departure_time_utc, route.quality or ""]
        ),
        capacity_cm3=np.array(route.capacity_cm3),
        stop_ids=np.array(route.stop_ids),
        zone_ids=np.array([zone or "" for zone in route.zone_ids]),
        lat=route.lat,
        lng=route.lng,
        times_tenths=tenths.astype(np.int32),
        n_packages=route.n_packages,
        service_time=route.service_time,
        window_start=route.window_start,
        window_end=route.window_end,
        actual=np.array(route.actual if route.actual is not None else [], dtype=np.int32),
    )
    return target


def load_route(path: Path) -> Route:
    with np.load(path) as data:
        route_id, station_code, date, departure, quality = data["meta"].tolist()
        actual = data["actual"].tolist()
        return Route(
            id=route_id,
            station_code=station_code,
            date=date,
            departure_time_utc=departure,
            capacity_cm3=float(data["capacity_cm3"]),
            quality=quality or None,
            stop_ids=data["stop_ids"].tolist(),
            lat=data["lat"],
            lng=data["lng"],
            zone_ids=[zone or None for zone in data["zone_ids"].tolist()],
            times=data["times_tenths"] / 10.0,
            n_packages=data["n_packages"],
            service_time=data["service_time"],
            window_start=data["window_start"],
            window_end=data["window_end"],
            actual=actual or None,
        )


def processed_files(split: str) -> list[Path]:
    """Sorted route files of a prepared split (``"train"`` or ``"eval"``)."""
    return sorted((paths.PROCESSED_DIR / split).glob("RouteID_*.npz"))


def iter_routes(split: str) -> Iterator[Route]:
    for path in processed_files(split):
        yield load_route(path)


def load_sample() -> list[Route]:
    """The 13 sample routes that ship with the training data, read from the raw files."""
    return list(
        iter_raw_routes(
            paths.SAMPLE_ROUTES,
            paths.SAMPLE_TRAVEL_TIMES,
            paths.SAMPLE_PACKAGES,
            paths.SAMPLE_ACTUAL,
        )
    )


def to_submission(routes: Sequence[Route], sequences: Sequence[Sequence[int]]) -> dict[str, dict]:
    """Convert proposed sequences to the official submission layout."""
    return {
        route.id: {"proposed": {route.stop_ids[stop]: i for i, stop in enumerate(sequence)}}
        for route, sequence in zip(routes, sequences)
    }
