"""Partition-before-augment split, and the leakage assertions that make it evidence.

Augmenting before splitting places near-duplicate variants of one source radiograph in
both the training and the test partition, so reported accuracy partly measures
memorisation of a specific image. Splitting first guarantees every source radiograph
lands in exactly one partition. ``verify_no_leakage`` is the programmatic check, run
both immediately after the split and again after augmentation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .preprocess import list_images


def stratified_split(paths_by_class: dict[str, list[Path]], train: float, val: float,
                     test: float, seed: int) -> dict[str, list[tuple[str, str]]]:
    """Split each class independently so partition proportions hold within every class.

    Two stages, matching the conventional ``train_test_split`` idiom: hold out the test
    fraction first, then carve the validation fraction out of what remains at the
    rescaled rate ``val / (1 - test)``. Each stage rounds the held-out count up. The
    ordering matters for the counts, not just the code: on 374 Osteopenia images it
    yields 261/56/57 rather than the 262/56/56 a single simultaneous rounding gives.
    """
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {train + val + test}")
    rng = np.random.RandomState(seed)
    out: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for class_name in sorted(paths_by_class):
        items = sorted(paths_by_class[class_name])
        n = len(items)
        index = rng.permutation(n)
        # round before ceil: 663 * (0.15/0.85) evaluates to 117.00000000000001 in
        # binary floating point, and a bare ceil would silently move an image.
        n_test = int(np.ceil(round(test * n, 9)))
        remaining = n - n_test
        n_val = int(np.ceil(round((val / (1.0 - test)) * remaining, 9)))
        chunks = {
            "test": index[:n_test],
            "val": index[n_test:n_test + n_val],
            "train": index[n_test + n_val:],
        }
        for partition, idx in chunks.items():
            out[partition].extend((str(items[i]), class_name) for i in sorted(idx))
    return out


def verify_no_leakage(split: dict[str, list[tuple[str, str]]]) -> dict[str, int]:
    """Assert zero filename-stem overlap between partitions.

    Stems rather than full paths: after offline augmentation an augmented file is named
    ``<stem>_aug<k>``, and comparing the source stem is what actually tests whether a
    variant of a test radiograph reached training.
    """
    stems = {
        part: {Path(p).stem.split("_aug")[0] for p, _ in items}
        for part, items in split.items()
    }
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = stems[a] & stems[b]
        overlaps[f"{a}_{b}"] = len(shared)
        if shared:
            raise AssertionError(
                f"LEAKAGE: {len(shared)} source stems appear in both {a} and {b}; "
                f"examples: {sorted(shared)[:5]}"
            )
    return overlaps


def class_counts(split: dict[str, list[tuple[str, str]]], classes: list[str]) -> dict:
    return {
        part: {c: sum(1 for _, lab in items if lab == c) for c in classes}
        for part, items in split.items()
    }


def build_split(raw_dir: str | Path, classes: list[str], fractions: dict[str, float],
                seed: int) -> dict:
    raw_dir = Path(raw_dir)
    paths_by_class = {c: list_images(raw_dir / c) for c in classes}
    missing = [c for c, v in paths_by_class.items() if not v]
    if missing:
        raise FileNotFoundError(f"no images found for class(es) {missing} under {raw_dir}")
    split = stratified_split(paths_by_class, fractions["train"], fractions["val"],
                             fractions["test"], seed)
    overlaps = verify_no_leakage(split)
    return {
        "split": split,
        "counts": class_counts(split, classes),
        "totals": {p: len(v) for p, v in split.items()},
        "source_counts": {c: len(v) for c, v in paths_by_class.items()},
        "leakage_check": overlaps,
        "seed": seed,
    }


def save_manifest(manifest: dict, path: str | Path) -> Path:
    """Write the partition manifest -- the file that makes the split auditable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def load_manifest(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
