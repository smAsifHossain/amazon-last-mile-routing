"""Convert the raw JSON files into one compact file per route.

    python scripts/prepare_data.py train
    python scripts/prepare_data.py eval

The raw files are read one route at a time, so this works on machines with far
less memory than the files would need when loaded whole. Output goes to
``data/processed/<split>/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lastmile import paths  # noqa: E402
from lastmile.data import iter_raw_routes, save_route  # noqa: E402

SPLITS = {
    "train": (paths.TRAIN_ROUTES, paths.TRAIN_TRAVEL_TIMES, paths.TRAIN_PACKAGES, paths.TRAIN_ACTUAL),
    "eval": (paths.EVAL_ROUTES, paths.EVAL_TRAVEL_TIMES, paths.EVAL_PACKAGES, paths.EVAL_ACTUAL),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("split", choices=sorted(SPLITS))
    args = parser.parse_args()

    sources = SPLITS[args.split]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        print("Missing raw files (run scripts/download_data.py first):\n  " + "\n  ".join(missing))
        return 1

    folder = paths.PROCESSED_DIR / args.split
    started = time.time()
    count = 0
    for count, route in enumerate(iter_raw_routes(*sources), start=1):
        save_route(route, folder)
        if count % 500 == 0:
            print(f"  {count} routes  ({time.time() - started:.0f}s)", flush=True)
    print(f"Done: {count} routes in {folder}  ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
