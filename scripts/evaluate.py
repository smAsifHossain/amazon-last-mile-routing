"""Score the model on routes it has not seen and write the app's data files.

    python scripts/evaluate.py                # the 3,072 official evaluation routes
    python scripts/evaluate.py --split heldout  # one fifth of the training routes

``eval`` fits the model on all 6,112 training routes and scores the challenge's
official evaluation routes (run ``prepare_data.py train`` and ``prepare_data.py
eval`` first). ``heldout`` fits on four fifths of the training routes and scores
the rest, which needs only the training data.

Writes two files for the web app in ``docs/data``:

* ``summary.json``: score of every scored route under each method;
* ``routes.json``: full detail for a selection of routes.

Takes about 20 minutes for ``eval`` on 4 cores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lastmile import baselines, model, paths, scoring, zones  # noqa: E402
from lastmile.data import load_route, processed_files  # noqa: E402
from validate import is_held_out, load_views  # noqa: E402

OUTPUT = paths.PROJECT_ROOT / "docs" / "data"
CITIES = {"DLA": "Los Angeles", "DSE": "Seattle", "DCH": "Chicago", "DBO": "Boston", "DAU": "Austin"}
METHODS = ["shortest", "fixed", "learned", "known"]
_MODEL: model.ZoneModel | None = None


def _init(fitted: model.ZoneModel) -> None:
    global _MODEL
    _MODEL = fitted


def zone_order(view: model.ZoneView, sequence: list[int]) -> list[int]:
    """Zones in the order a sequence first reaches them."""
    return list(dict.fromkeys(int(view.stop_zone[stop]) for stop in sequence[1:]))


def process(task: tuple[Path, bool]) -> tuple[dict, dict | None]:
    path, with_detail = task
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
    scores = {name: round(scoring.score_sequence(route.actual, sequences[name], route.times), 4) for name in METHODS}
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
    if not with_detail:
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


def spread(files: list[Path], count: int) -> set[Path]:
    """Routes for the explorer, taken in turn from each station (not chosen by result)."""
    by_station = defaultdict(list)
    for path in files:
        with np.load(path) as data:
            by_station[str(data["meta"][1])].append(path)
    chosen: list[Path] = []
    rank = 0
    while len(chosen) < count and any(len(group) > rank for group in by_station.values()):
        for station in sorted(by_station):
            if len(by_station[station]) > rank and len(chosen) < count:
                chosen.append(by_station[station][rank])
        rank += 1
    return set(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["eval", "heldout"], default="eval")
    parser.add_argument("--routes", type=int, default=60, help="routes with full detail for the explorer")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    views = load_views(args.workers)
    if not views:
        print("No prepared training routes. Run scripts/prepare_data.py train first.")
        return 1
    if args.split == "eval":
        fit_views = [view for _, _, view in views]
        targets = processed_files("eval")
        label = "official evaluation routes"
        if not targets:
            print("No prepared evaluation routes. Run scripts/prepare_data.py eval first.")
            return 1
    else:
        fit_views = [view for _, _, view in views if not is_held_out(view.route_id)]
        targets = [path for path, _, view in views if is_held_out(view.route_id)]
        label = "held-out training routes"

    started = time.time()
    fitted = model.ZoneModel.fit(fit_views)
    print(f"Fitted on {len(fit_views)} training routes ({time.time() - started:.0f}s)")
    detail_paths = spread(targets, args.routes)

    print(f"Scoring {len(targets)} {label} ...", flush=True)
    started = time.time()
    summaries, details = [], []
    tasks = [(path, path in detail_paths) for path in targets]
    with Pool(args.workers, initializer=_init, initargs=(fitted,)) as pool:
        for summary, detail in pool.imap(process, tasks, chunksize=4):
            summaries.append(summary)
            if detail:
                details.append(detail)
            if len(summaries) % 200 == 0:
                print(f"  {len(summaries)}  ({time.time() - started:.0f}s)", flush=True)

    names = {"shortest": "Shortest route", "fixed": "Fixed zone penalties", "learned": "Learned zone order", "known": "Driver's zone order given"}
    print(f"\nMean score on {len(summaries)} {label} (lower is better):")
    for method in METHODS:
        print(f"  {names[method]:<28}{np.mean([s[method] for s in summaries]):.4f}")
    print(f"  first zone right            {np.mean([s['first_zone_right'] for s in summaries]):.1%}")
    print(f"  zone moves right            {np.mean([s['moves_right'] for s in summaries]):.1%}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    meta = {"split": args.split, "label": label, "fitted_on": len(fit_views), "scored": len(summaries), "routes": summaries}
    (OUTPUT / "summary.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    (OUTPUT / "routes.json").write_text(json.dumps(details, separators=(",", ":")), encoding="utf-8")
    for name in ("summary.json", "routes.json"):
        print(f"Wrote {OUTPUT / name} ({(OUTPUT / name).stat().st_size / 1024:.0f} KB)")
    print(f"{len(details)} routes with detail, {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
