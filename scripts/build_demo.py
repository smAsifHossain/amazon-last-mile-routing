"""Precompute the routes shown in the demo page (``docs/``).

    python scripts/build_demo.py

Picks held-out training routes spread over the stations, fits the model on the
other routes, and stores the driver's sequence next to the shortest route and
the model's route in ``docs/data/routes.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lastmile import baselines, model, paths, scoring  # noqa: E402
from lastmile.data import load_route  # noqa: E402
from validate import is_held_out, load_views  # noqa: E402

OUTPUT = paths.PROJECT_ROOT / "docs" / "data" / "routes.json"
CITIES = {"DLA": "Los Angeles", "DSE": "Seattle", "DCH": "Chicago", "DBO": "Boston", "DAU": "Austin"}


def pick(views, count: int) -> list[Path]:
    """Held-out high-quality routes, taken in turn from each station."""
    by_station = defaultdict(list)
    for path, quality, view in views:
        if is_held_out(view.route_id) and quality == "High":
            by_station[view.station_code].append(path)
    chosen: list[Path] = []
    rank = 0
    while len(chosen) < count and any(len(paths_) > rank for paths_ in by_station.values()):
        for station in sorted(by_station):
            if len(by_station[station]) > rank and len(chosen) < count:
                chosen.append(by_station[station][rank])
        rank += 1
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--routes", type=int, default=34)
    args = parser.parse_args()

    views = load_views(workers=3)
    fitted = model.ZoneModel.fit([view for _, _, view in views if not is_held_out(view.route_id)])

    demo = []
    for path in pick(views, args.routes):
        route = load_route(path)
        view = model.zone_view(route)
        sequences = {
            "driver": route.actual,
            "shortest": baselines.shortest_route(route),
            "learned": fitted.predict(route),
        }
        demo.append(
            {
                "id": route.id[8:16],
                "station": route.station_code,
                "city": CITIES.get(route.station_code[:3], route.station_code[:3]),
                "date": route.date,
                "zones": view.zones,
                "stops": [
                    [round(float(route.lat[i]), 5), round(float(route.lng[i]), 5), int(view.stop_zone[i])]
                    for i in range(route.n_stops)
                ],
                "sequences": sequences,
                "scores": {
                    name: round(scoring.score_sequence(route.actual, seq, route.times), 4)
                    for name, seq in sequences.items()
                },
                "minutes": {
                    name: round(baselines.tour_time(route, seq) / 60) for name, seq in sequences.items()
                },
            }
        )
        print(f"  {demo[-1]['station']} {demo[-1]['id']}  {demo[-1]['scores']}", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(demo, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(demo)} routes to {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
