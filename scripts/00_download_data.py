"""Fetch the Multi-Class Knee Osteoporosis X-Ray Dataset into data/raw/.

The corpus is a public Kaggle dataset and is not redistributed in this repository.
On Kaggle the dataset is already mounted, so this script only locates it.

    python scripts/00_download_data.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

DATASET = "mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset"
CLASS_DIR = "OS Collected Data"
CLASSES = ("Normal", "Osteopenia", "Osteoporosis")
EXPECTED = {"Normal": 780, "Osteopenia": 374, "Osteoporosis": 793}


def find_mounted() -> Path | None:
    """Locate the dataset if it is already attached (Kaggle) or cached locally."""
    candidates = [
        Path("/kaggle/input/datasets/mohamedgobara") /
        "multi-class-knee-osteoporosis-x-ray-dataset" / CLASS_DIR,
        Path("/kaggle/input/multi-class-knee-osteoporosis-x-ray-dataset") / CLASS_DIR,
        Path.home() / ".cache" / "kagglehub" / "datasets" / "mohamedgobara" /
        "multi-class-knee-osteoporosis-x-ray-dataset" / "versions" / "1" / CLASS_DIR,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    for root in (Path("/kaggle/input"),):
        if root.is_dir():
            for path in root.rglob(CLASS_DIR):
                if path.is_dir():
                    return path
    return None


def download() -> Path:
    import kagglehub
    root = Path(kagglehub.dataset_download(DATASET))
    found = root / CLASS_DIR
    if found.is_dir():
        return found
    matches = list(root.rglob(CLASS_DIR))
    if not matches:
        raise FileNotFoundError(f"{CLASS_DIR!r} not found under {root}")
    return matches[0]


def verify(path: Path) -> dict[str, int]:
    counts = {}
    for name in CLASSES:
        class_dir = path / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class directory: {class_dir}")
        counts[name] = sum(1 for p in class_dir.iterdir() if p.is_file())
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(REPO_ROOT / "data" / "raw" / CLASS_DIR))
    parser.add_argument("--link-only", action="store_true",
                        help="report the mounted location instead of copying")
    args = parser.parse_args()

    source = find_mounted() or download()
    counts = verify(source)
    total = sum(counts.values())
    print(f"source: {source}")
    for name, count in counts.items():
        flag = "OK" if count == EXPECTED[name] else f"EXPECTED {EXPECTED[name]}"
        print(f"  {name:14s} {count:5d}  {flag}")
    print(f"  {'TOTAL':14s} {total:5d}  {'OK' if total == 1947 else 'EXPECTED 1947'}")

    if args.link_only:
        print(f"\nuse --raw-dir '{source}' with scripts/01")
        return 0

    dest = Path(args.dest)
    if dest.resolve() == source.resolve():
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    for name in CLASSES:
        target = dest / name
        if target.exists():
            print(f"  {target} exists, skipping copy")
            continue
        shutil.copytree(source / name, target)
    print(f"\ncopied to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
