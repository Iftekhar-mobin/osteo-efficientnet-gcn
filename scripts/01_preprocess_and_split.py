"""Preprocess, split before augmenting, rebalance the training partition only.

Produces, in order:
  1. the class-wise bone-intensity analysis that motivated the CLAHE parameters
     (results/table_intensity.csv, figures/bone_intensity_distribution.png)
  2. the stratified 70/15/15 partition manifest, with the leakage assertion
     (data/splits/manifest.json, results/table_dataset_split.csv)
  3. the preprocessed 320x320 images and the offline class-balancing augmentation
     (data/processed/{train,val,test}, *_index.json)

    python scripts/01_preprocess_and_split.py --raw-dir "<path to OS Collected Data>"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import _bootstrap  # noqa: F401
from _bootstrap import FIGURES, REPO_ROOT, RESULTS

from osteognn.config import load_config
from osteognn.data.augment import balance_partition, build_offline_pipeline
from osteognn.data.preprocess import (bone_intensity_stats, list_images,
                                      preprocess_image)
from osteognn.data.split import build_split, save_manifest, verify_no_leakage
from osteognn.utils import save_json, set_seed

import cv2


def intensity_analysis(raw_dir: Path, classes: list[str]) -> dict:
    """Class-wise intensity means/SDs and a Kruskal--Wallis test across the groups.

    A non-parametric test is used because per-image mean intensity is not normally
    distributed across a radiograph corpus with heterogeneous exposure settings.
    """
    per_class, samples_whole, samples_bone = {}, [], []
    for name in classes:
        stats_dict = bone_intensity_stats(list_images(raw_dir / name))
        samples_whole.append(stats_dict.pop("_per_image_whole"))
        samples_bone.append(stats_dict.pop("_per_image_bone"))
        per_class[name] = stats_dict
    h_whole, p_whole = stats.kruskal(*samples_whole)
    h_bone, p_bone = stats.kruskal(*samples_bone)
    epsilon_sq = (h_bone - len(classes) + 1) / (sum(len(s) for s in samples_bone) - len(classes))
    return {
        "per_class": per_class,
        "kruskal_whole": {"H": float(h_whole), "p": float(p_whole),
                          "df": len(classes) - 1},
        "kruskal_bone": {"H": float(h_bone), "p": float(p_bone),
                         "df": len(classes) - 1, "epsilon_squared": float(epsilon_sq)},
        "_samples_whole": [s.tolist() for s in samples_whole],
        "_samples_bone": [s.tolist() for s in samples_bone],
    }


def write_processed(plan: list[tuple[str, str, int]], out_dir: Path, cfg,
                    pipeline=None, seed: int = 42) -> list[dict]:
    """Materialise preprocessed (and, where required, augmented) images to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    index: list[dict] = []
    for source, label, aug_index in plan:
        class_dir = out_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem.replace(" ", "_")
        name = f"{stem}.png" if aug_index == 0 else f"{stem}_aug{aug_index}.png"
        target = class_dir / name
        if not target.exists():
            image = preprocess_image(
                source, size=int(cfg.preprocess.image_size),
                clahe=bool(cfg.preprocess.clahe),
                clip_limit=float(cfg.preprocess.clahe_clip_limit),
                tile_grid=int(cfg.preprocess.clahe_tile_grid))
            if aug_index > 0 and pipeline is not None:
                pipeline.set_random_seed = None  # albumentations uses the global RNG
                np.random.seed(int(rng.randint(0, 2 ** 31 - 1)))
                image = pipeline(image=image)["image"]
            cv2.imwrite(str(target), image[:, :, 0])
        index.append({"path": str(target), "label": label, "source": source,
                      "augmented": aug_index > 0})
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--skip-intensity", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.raw_dir:
        cfg.set_path("data.raw_dir", args.raw_dir)
    set_seed(int(cfg.seed))
    raw_dir = Path(cfg.data.raw_dir)
    classes = list(cfg.data.classes)
    tag = "" if cfg.preprocess.clahe else "_noclahe"
    processed_root = Path(str(cfg.data.processed_dir) + tag)
    cfg.set_path("data.processed_dir", str(processed_root))

    # ---- 1. bone-intensity analysis -------------------------------------------
    if not args.skip_intensity:
        print("[1/3] bone-intensity analysis", flush=True)
        analysis = intensity_analysis(raw_dir, classes)
        save_json({k: v for k, v in analysis.items() if not k.startswith("_")},
                  RESULTS / "intensity_analysis.json")
        with open(RESULTS / "table_intensity.csv", "w", encoding="utf-8") as handle:
            handle.write("class,n,whole_mean,whole_sd,bone_mean,bone_sd\n")
            for name in classes:
                s = analysis["per_class"][name]
                handle.write(f"{name},{s['n']},{s['mean']:.2f},{s['sd']:.2f},"
                             f"{s['bone_mean']:.2f},{s['bone_sd']:.2f}\n")
        np.savez(RESULTS / "intensity_samples.npz",
                 **{f"whole_{c}": np.array(s) for c, s in
                    zip(classes, analysis["_samples_whole"])},
                 **{f"bone_{c}": np.array(s) for c, s in
                    zip(classes, analysis["_samples_bone"])})
        for name in classes:
            s = analysis["per_class"][name]
            print(f"    {name:14s} bone mean {s['bone_mean']:6.2f} "
                  f"+/- {s['bone_sd']:5.2f}   (whole {s['mean']:6.2f})")
        print(f"    Kruskal-Wallis (bone): H={analysis['kruskal_bone']['H']:.2f}, "
              f"p={analysis['kruskal_bone']['p']:.3e}")

    # ---- 2. split BEFORE augmentation -----------------------------------------
    print("[2/3] stratified 70/15/15 split (before any augmentation)", flush=True)
    manifest = build_split(raw_dir, classes,
                           {"train": float(cfg.split.train), "val": float(cfg.split.val),
                            "test": float(cfg.split.test)}, int(cfg.seed))
    splits_dir = Path(cfg.data.splits_dir)
    save_manifest(manifest, splits_dir / "manifest.json")
    for partition, counts in manifest["counts"].items():
        print(f"    {partition:5s} {counts}  total={manifest['totals'][partition]}")
    print(f"    leakage check (shared source stems): {manifest['leakage_check']}")

    # ---- 3. preprocess + training-only balancing ------------------------------
    print("[3/3] preprocessing and training-partition balancing", flush=True)
    pipeline = build_offline_pipeline(cfg.augment) if cfg.augment.balance else None
    summary = {}
    for partition in ("train", "val", "test"):
        items = [(p, lab) for p, lab in manifest["split"][partition]]
        if partition == "train" and cfg.augment.balance:
            plan = balance_partition(items, classes,
                                     int(cfg.augment.target_per_class), int(cfg.seed))
        else:
            plan = [(p, lab, 0) for p, lab in items]
        index = write_processed(plan, processed_root / partition, cfg,
                                pipeline if partition == "train" else None,
                                seed=int(cfg.seed))
        with open(processed_root / f"{partition}_index.json", "w", encoding="utf-8") as h:
            json.dump(index, h, indent=2)
        counts = {c: sum(1 for r in index if r["label"] == c) for c in classes}
        summary[partition] = {"counts": counts, "total": len(index),
                              "augmented": sum(1 for r in index if r["augmented"])}
        print(f"    {partition:5s} {counts}  total={len(index)} "
              f"(augmented {summary[partition]['augmented']})")

    # Re-assert leakage freedom on the materialised files, augmentation included.
    materialised = {
        part: [(r["path"], r["label"])
               for r in json.load(open(processed_root / f"{part}_index.json",
                                       encoding="utf-8"))]
        for part in ("train", "val", "test")
    }
    post = verify_no_leakage(materialised)
    print(f"    post-augmentation leakage check: {post}")

    with open(RESULTS / "table_dataset_split.csv", "w", encoding="utf-8") as handle:
        handle.write("class,train_orig,val,test,train_aug\n")
        for name in classes:
            handle.write(f"{name},{manifest['counts']['train'][name]},"
                         f"{manifest['counts']['val'][name]},"
                         f"{manifest['counts']['test'][name]},"
                         f"{summary['train']['counts'][name]}\n")
        handle.write(f"Total,{manifest['totals']['train']},{manifest['totals']['val']},"
                     f"{manifest['totals']['test']},{summary['train']['total']}\n")

    save_json({"manifest_counts": manifest["counts"], "totals": manifest["totals"],
               "processed": summary, "leakage_after_split": manifest["leakage_check"],
               "leakage_after_augmentation": post,
               "processed_dir": str(processed_root)},
              RESULTS / "dataset_summary.json")
    print(f"\nwrote {processed_root} and {RESULTS/'dataset_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
