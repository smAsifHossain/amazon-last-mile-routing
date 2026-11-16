"""Loading challenge routes into plain Python objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from . import paths


@dataclass(frozen=True)
class Stop:
    id: str
    lat: float
    lng: float
    is_station: bool
    zone_id: str | None  # missing for the station and for some drop-offs


@dataclass
class Route:
    id: str
    station_code: str
    date: str
    departure_time_utc: str
    capacity_cm3: float
    quality: str | None  # "High" / "Medium" / "Low"; only in the training set
    stops: dict[str, Stop]
    travel_times: dict[str, dict[str, float]]  # seconds, [origin][destination]
    packages: dict[str, dict] = field(default_factory=dict, repr=False)
    actual: list[str] | None = None  # driver's sequence, starting at the station

    @property
    def station(self) -> str:
        return next(stop.id for stop in self.stops.values() if stop.is_station)

    @property
    def dropoffs(self) -> list[str]:
        return [stop.id for stop in self.stops.values() if not stop.is_station]


def _read(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)  # the files use NaN for missing values


def load_routes(
    routes_path: Path,
    travel_times_path: Path,
    packages_path: Path | None = None,
    actual_path: Path | None = None,
) -> dict[str, Route]:
    """Load routes keyed by route id. Reads whole files into memory."""
    raw_routes = _read(routes_path)
    travel_times = _read(travel_times_path)
    packages = _read(packages_path) if packages_path else {}
    actual = _read(actual_path) if actual_path else {}

    routes: dict[str, Route] = {}
    for route_id, raw in raw_routes.items():
        stops = {
            stop_id: Stop(
                id=stop_id,
                lat=s["lat"],
                lng=s["lng"],
                is_station=s["type"] == "Station",
                zone_id=s["zone_id"] if isinstance(s["zone_id"], str) else None,
            )
            for stop_id, s in raw["stops"].items()
        }
        sequence = None
        if route_id in actual:
            positions = actual[route_id]["actual"]
            sequence = sorted(positions, key=positions.__getitem__)
        routes[route_id] = Route(
            id=route_id,
            station_code=raw["station_code"],
            date=raw["date_YYYY_MM_DD"],
            departure_time_utc=raw["departure_time_utc"],
            capacity_cm3=raw["executor_capacity_cm3"],
            quality=raw.get("route_score"),
            stops=stops,
            travel_times=travel_times[route_id],
            packages=packages.get(route_id, {}),
            actual=sequence,
        )
    return routes


def load_sample() -> dict[str, Route]:
    """The 13 sample routes that ship with the training data."""
    return load_routes(
        paths.SAMPLE_ROUTES,
        paths.SAMPLE_TRAVEL_TIMES,
        paths.SAMPLE_PACKAGES,
        paths.SAMPLE_ACTUAL,
    )


def to_submission(sequences: dict[str, Sequence[str]]) -> dict[str, dict]:
    """Convert ``{route_id: [station, stop, ...]}`` to the official file layout."""
    return {
        route_id: {"proposed": {stop: i for i, stop in enumerate(sequence)}}
        for route_id, sequence in sequences.items()
    }
