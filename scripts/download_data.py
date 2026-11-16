"""Download the 2021 Amazon Last Mile Routing Research Challenge dataset.

The data is public on the AWS Registry of Open Data and needs no account:
https://registry.opendata.aws/amazon-last-mile-challenges

Usage::

    python scripts/download_data.py --list            # show files and sizes
    python scripts/download_data.py --subset sample   # a handful of routes (~5 MB)
    python scripts/download_data.py --subset train    # 6,112 routes (~2.3 GB)
    python scripts/download_data.py --subset eval     # 3,072 routes (~1.1 GB)
    python scripts/download_data.py                   # everything (~3.3 GB)

Files land in ``data/raw/``. Files already present with the right size are skipped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET_URL = "https://amazon-last-mile-challenges.s3.amazonaws.com"
PREFIX = "almrrc2021/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

SUBSETS = {
    "all": lambda key: True,
    "train": lambda key: "/almrrc2021-data-training/" in key,
    "eval": lambda key: "/almrrc2021-data-evaluation/" in key,
    "sample": lambda key: "/almrrc2021-data-training/" in key and "/new_" in key,
}


def list_files() -> list[tuple[str, int]]:
    """Return ``(key, size_in_bytes)`` for every file in the dataset."""
    files: list[tuple[str, int]] = []
    token = None
    while True:
        query = {"list-type": "2", "prefix": PREFIX}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET_URL}/?{urllib.parse.urlencode(query)}") as response:
            root = ET.fromstring(response.read())
        for item in root.iter(f"{S3_NS}Contents"):
            key = item.findtext(f"{S3_NS}Key")
            size = int(item.findtext(f"{S3_NS}Size"))
            if not key.endswith("/"):
                files.append((key, size))
        if root.findtext(f"{S3_NS}IsTruncated") != "true":
            return files
        token = root.findtext(f"{S3_NS}NextContinuationToken")


def download(key: str, size: int, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    url = f"{BUCKET_URL}/{urllib.parse.quote(key)}"
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        done = 0
        next_report = 0.1
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if size and done / size >= next_report:
                print(f"    {done / size:4.0%}", flush=True)
                next_report += 0.1
    if partial.stat().st_size != size:
        raise IOError(f"{key}: expected {size} bytes, got {partial.stat().st_size}")
    shutil.move(partial, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subset", choices=sorted(SUBSETS), default="all")
    parser.add_argument("--list", action="store_true", help="only list the files")
    args = parser.parse_args()

    files = [(key, size) for key, size in list_files() if SUBSETS[args.subset](key)]
    total = sum(size for _, size in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB ({args.subset})")
    if args.list:
        for key, size in files:
            print(f"  {size / 1e6:9.1f} MB  {key[len(PREFIX):]}")
        return 0

    for key, size in files:
        target = RAW_DIR / key[len(PREFIX):]
        if target.exists() and target.stat().st_size == size:
            print(f"  have  {target.relative_to(RAW_DIR)}")
            continue
        print(f"  get   {target.relative_to(RAW_DIR)}  ({size / 1e6:.1f} MB)", flush=True)
        download(key, size, target)
    print(f"Done. Data is in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
