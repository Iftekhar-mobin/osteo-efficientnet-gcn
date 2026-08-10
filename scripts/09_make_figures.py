"""Regenerate every measured figure from results/.

Figures are written to figures/ and, where the manuscript includes them by name, also
to the repository root so paper.tex compiles unchanged.

    python scripts/09_make_figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import FIGURES, REPO_ROOT, RESULTS

from osteognn import figures as F
from osteognn.config import load_config
from osteognn.data.preprocess import preprocess_image, read_grayscale
from osteognn.data.split import load_manifest
from osteognn.utils import load_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run", default="full")
    args = parser.parse_args()
    cfg = load_config(args.config)
    made = []

    run_path = RESULTS / f"run_{args.run}.json"
    if run_path.exists():
        run = load_json(run_path)
        made.append(F.training_curves(run["training"]["history"],
                                      FIGURES / "training_curves.png",
                                      REPO_ROOT / "gcn_training_curves.png"))
        made.append(F.roc_curves(run["test"], FIGURES / "roc_curves.png",
                                 REPO_ROOT / "gcn_roc_curves.png"))
        made.append(F.confusion_matrix(run["test"], FIGURES / "confusion_matrix.png",
                                       REPO_ROOT / "gcn_confusion_matrix.png"))
        made.append(F.reliability_diagram(run["test"], FIGURES / "reliability.png"))

    intensity_path = RESULTS / "intensity_analysis.json"
    samples_path = RESULTS / "intensity_samples.npz"
    if intensity_path.exists() and samples_path.exists():
        made.append(F.intensity_distribution(
            dict(np.load(samples_path)), list(cfg.data.classes),
            load_json(intensity_path), FIGURES / "bone_intensity_distribution.png",
            REPO_ROOT / "bone_intensity_distribution.png"))

    manifest_path = Path(cfg.data.splits_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        pairs = []
        for name in list(cfg.data.classes):
            source = next(p for p, lab in manifest["split"]["test"] if lab == name)
            pairs.append((name, read_grayscale(source),
                          preprocess_image(source, int(cfg.preprocess.image_size),
                                           bool(cfg.preprocess.clahe),
                                           float(cfg.preprocess.clahe_clip_limit),
                                           int(cfg.preprocess.clahe_tile_grid))[:, :, 0]))
        made.append(F.preprocessing_comparison(
            pairs, FIGURES / "preprocessed_comparison.png",
            REPO_ROOT / "preprocessed_comparison.png"))

    for name, builder, out in (
        ("ablation.json", F.ablation_bars, "ablation_deltas.png"),
        ("fusion_sweep.json", F.fusion_sweep, "fusion_sweep.png"),
        ("multiseed.json", F.multiseed_spread, "multiseed_spread.png"),
    ):
        path = RESULTS / name
        if path.exists():
            made.append(builder(load_json(path), FIGURES / out))

    for path in made:
        print(f"  {path}")
    print(f"\n{len(made)} figures written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
