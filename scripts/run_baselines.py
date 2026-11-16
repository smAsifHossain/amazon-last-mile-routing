"""Score each sequencing method on the 13 sample routes.

    python scripts/run_baselines.py

Scores follow the official challenge metric: lower is better, 0 is a perfect
match with the driver's sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lastmile import baselines, scoring, zones  # noqa: E402
from lastmile.data import load_sample  # noqa: E402

METHODS = {
    "random": baselines.random_order,
    "nearest": baselines.nearest_neighbour,
    "shortest": zones.shortest_route,
    "zone-aware": zones.zone_aware_route,
}


def main() -> None:
    routes = load_sample()
    print(f"{len(routes)} sample routes\n")
    print(f"{'route':<14}{'station':<9}{'stops':>6}" + "".join(f"{name:>12}" for name in METHODS))
    totals = {name: 0.0 for name in METHODS}
    for route in routes.values():
        actual = route.actual + [route.actual[0]]
        line = f"{route.id[8:20]:<14}{route.station_code:<9}{len(route.dropoffs):>6}"
        for name, method in METHODS.items():
            sequence = method(route)
            score = scoring.score_route(actual, sequence + [sequence[0]], route.travel_times)
            totals[name] += score
            line += f"{score:>12.4f}"
        print(line, flush=True)
    print(f"{'mean':<29}" + "".join(f"{totals[name] / len(routes):>12.4f}" for name in METHODS))


if __name__ == "__main__":
    main()
