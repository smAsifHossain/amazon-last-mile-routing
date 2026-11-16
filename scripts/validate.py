"""Fit the zone model on part of the training routes and score it on the rest.

    python scripts/prepare_data.py train      # once
    python scripts/validate.py                # all held-out routes
    python scripts/validate.py --limit 300    # quicker check

One route in five (chosen by a hash of its id) is held out. The model never sees
those routes, neither for its weights nor for its zone history. Scores use the
official challenge metric: lower is better.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lastmile import baselines, model, paths, scoring, zones  # noqa: E402
from lastmile.data import load_route, processed_files  # noqa: E402

VIEW_CACHE = paths.PROCESSED_DIR / "train_zone_views.pkl"
_MODEL: model.ZoneModel | None = None


def is_held_out(route_id: str) -> bool:
    return int(hashlib.md5(route_id.encode()).hexdigest(), 16) % 5 == 0


def view_of(path: Path):
    route = load_route(path)
    return path, route.quality, model.zone_view(route)


def load_views(workers: int):
    """Zone view of every training route, cached on disk after the first run."""
    files = processed_files("train")
    if VIEW_CACHE.exists():
        with open(VIEW_CACHE, "rb") as handle:
            cached = pickle.load(handle)
        if len(cached) == len(files):
            return cached
    print(f"Reading {len(files)} routes ...", flush=True)
    with Pool(workers) as pool:
        views = list(pool.imap(view_of, files, chunksize=50))
    with open(VIEW_CACHE, "wb") as handle:
        pickle.dump(views, handle)
    return views


def _init(fitted: model.ZoneModel) -> None:
    global _MODEL
    _MODEL = fitted


def score_route(path: Path) -> dict:
    route = load_route(path)
    view = model.zone_view(route)
    sequences = {
        "shortest": baselines.shortest_route(route),
        "fixed": zones.zone_aware_route(route),
        "learned": _MODEL.predict(route),
        "known": model.sequence_by_zones(route, view, view.order),
    }
    result = {name: scoring.score_sequence(route.actual, seq, route.times) for name, seq in sequences.items()}
    result["quality"] = route.quality
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=0, help="score only this many held-out routes")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    views = load_views(args.workers)
    if not views:
        print("No prepared routes. Run scripts/prepare_data.py train first.")
        return 1
    fit_views = [view for path, _, view in views if not is_held_out(view.route_id)]
    held_out = [path for path, _, view in views if is_held_out(view.route_id)]
    if args.limit:
        held_out = held_out[: args.limit]

    started = time.time()
    fitted = model.ZoneModel.fit(fit_views)
    print(f"Fitted on {len(fit_views)} routes in {time.time() - started:.0f}s")
    np.set_printoptions(precision=2, suppress=True)
    print("  next-zone weights ", fitted.w_pair)
    print("  first-zone weights", fitted.w_start)
    print("  last-zone weights ", fitted.w_end)

    print(f"Scoring {len(held_out)} held-out routes ...", flush=True)
    started = time.time()
    results = []
    with Pool(args.workers, initializer=_init, initargs=(fitted,)) as pool:
        for result in pool.imap_unordered(score_route, held_out, chunksize=4):
            results.append(result)
            if len(results) % 200 == 0:
                print(f"  {len(results)}  ({time.time() - started:.0f}s)", flush=True)

    methods = ["shortest", "fixed", "learned", "known"]
    print(f"\n{'routes':<16}{'count':>6}" + "".join(f"{name:>10}" for name in methods))
    for label, keep in [("all", None), ("High", "High"), ("Medium", "Medium"), ("Low", "Low")]:
        rows = [r for r in results if keep is None or r["quality"] == keep]
        if rows:
            means = "".join(f"{np.mean([r[name] for r in rows]):>10.4f}" for name in methods)
            print(f"{label:<16}{len(rows):>6}{means}")
    print(f"\n{time.time() - started:.0f}s. 'known' uses the driver's own zone order and shows the")
    print("best this approach could reach if the zone order were predicted perfectly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
