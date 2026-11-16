"""Precompute the data behind the demo app (``docs/``).

    python scripts/build_demo.py

Fits the model on four fifths of the training routes and writes two files:

* ``docs/data/summary.json``: score of every held-out route under each method;
* ``docs/data/routes.json``: full detail for a selection of held-out routes
  (stops, zones, and the driver's, shortest and learned sequences).

Takes about 10 minutes on 4 cores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lastmile import baselines, model, paths, scoring, zones  # noqa: E402
from lastmile.data import load_route  # noqa: E402
from validate import is_held_out, load_views  # noqa: E402

OUTPUT = paths.PROJECT_ROOT / "docs" / "data"
CITIES = {"DLA": "Los Angeles", "DSE": "Seattle", "DCH": "Chicago", "DBO": "Boston", "DAU": "Austin"}
_MODEL: model.ZoneModel | None = None
_DETAIL: set[str] = set()


def _init(fitted: model.ZoneModel, detail: set[str]) -> None:
    global _MODEL, _DETAIL
    _MODEL, _DETAIL = fitted, detail


def pick(views, count: int) -> set[str]:
    """Ids of held-out routes for the explorer, taken in turn from each station."""
    by_station = defaultdict(list)
    for _, quality, view in views:
        if is_held_out(view.route_id) and quality in ("High", "Medium"):
            by_station[view.station_code].append(view.route_id)
    chosen: list[str] = []
    rank = 0
    while len(chosen) < count and any(len(ids) > rank for ids in by_station.values()):
        for station in sorted(by_station):
            if len(by_station[station]) > rank and len(chosen) < count:
                chosen.append(by_station[station][rank])
        rank += 1
    return set(chosen)


def zone_order(view: model.ZoneView, sequence: list[int]) -> list[int]:
    """Zones in the order a sequence first reaches them."""
    return list(dict.fromkeys(int(view.stop_zone[stop]) for stop in sequence[1:]))


def process(path: Path) -> tuple[dict, dict | None]:
    route = load_route(path)
    view = model.zone_view(route)
    learned_order = _MODEL.zone_order(view)
    sequences = {
        "driver": route.actual,
        "shortest": baselines.shortest_route(route),
        "fixed": zones.zone_aware_route(route),
        "learned": model.sequence_by_zones(route, view, learned_order),
        "known": model.sequence_by_zones(route, view, view.order),
    }
    scores = {
        name: round(scoring.score_sequence(route.actual, seq, route.times), 4)
        for name, seq in sequences.items()
        if name != "driver"
    }
    k = len(view.zones)
    moves = set(zip(view.order, view.order[1:]))
    summary = {
        "id": route.id[8:16],
        "city": CITIES.get(route.station_code[:3], route.station_code[:3]),
        "station": route.station_code,
        "quality": route.quality,
        "stops": route.n_stops - 1,
        "zones": k,
        "first_zone_right": learned_order[0] == view.order[0],
        "moves_right": round(len(moves & set(zip(learned_order, learned_order[1:]))) / max(k - 1, 1), 3),
        **scores,
    }
    if route.id not in _DETAIL:
        return summary, None

    shown = {name: sequences[name] for name in ("driver", "shortest", "learned")}
    detail = {
        **summary,
        "date": route.date,
        "zone_names": view.zones,
        "coords": [[round(float(route.lat[i]), 5), round(float(route.lng[i]), 5)] for i in range(route.n_stops)],
        "stop_zone": view.stop_zone.tolist(),
        "packages": route.n_packages.tolist(),
        "sequences": shown,
        "zone_orders": {name: zone_order(view, seq) for name, seq in shown.items()},
        # Seconds of driving for each leg of a sequence, including the return.
        "legs": {
            name: [round(float(route.times[a, b])) for a, b in zip(seq, seq[1:] + seq[:1])]
            for name, seq in shown.items()
        },
    }
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--routes", type=int, default=51, help="routes with full detail")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    views = load_views(args.workers)
    fit_views = [view for _, _, view in views if not is_held_out(view.route_id)]
    held_out = [path for path, _, view in views if is_held_out(view.route_id)]
    fitted = model.ZoneModel.fit(fit_views)
    detail_ids = pick(views, args.routes)

    print(f"Scoring {len(held_out)} held-out routes ...", flush=True)
    started = time.time()
    summaries, details = [], []
    with Pool(args.workers, initializer=_init, initargs=(fitted, detail_ids)) as pool:
        for summary, detail in pool.imap(process, held_out, chunksize=4):
            summaries.append(summary)
            if detail:
                details.append(detail)
            if len(summaries) % 200 == 0:
                print(f"  {len(summaries)}  ({time.time() - started:.0f}s)", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    meta = {"fitted_on": len(fit_views), "held_out": len(summaries), "routes": summaries}
    (OUTPUT / "summary.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    (OUTPUT / "routes.json").write_text(json.dumps(details, separators=(",", ":")), encoding="utf-8")
    for name in ("summary.json", "routes.json"):
        print(f"Wrote {OUTPUT / name} ({(OUTPUT / name).stat().st_size / 1024:.0f} KB)")
    print(f"{len(summaries)} routes scored, {len(details)} with detail, {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
