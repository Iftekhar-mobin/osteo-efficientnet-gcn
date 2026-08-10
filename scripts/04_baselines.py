"""Baselines under the identical protocol, plus the partition-ordering probe.

Cross-study accuracy comparisons confound architecture with dataset, class count and
partitioning protocol. Every model here is trained and evaluated under exactly the
preprocessing, partition, augmentation and training budget of the proposed method, so
the only thing that varies is the architecture.

The probe is separate and answers a different question: how much of the margin reported
by studies that augment *before* partitioning is produced by that ordering alone.

    python scripts/04_baselines.py
    python scripts/04_baselines.py --only vgg19 --probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import CHECKPOINTS, REPO_ROOT, RESULTS

from osteognn.config import BASELINES, load_config, variant_config
from osteognn.data.datasets import build_loaders
from osteognn.metrics import mcnemar_test
from osteognn.train import train_model
from osteognn.utils import load_json, save_json, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_train_script = import_module("02_train")

ORDER = ["effb0_gap_fc", "resnet50", "densenet121", "vgg19"]
LABELS = {
    "effb0_gap_fc": "EfficientNetB0 + GAP + FC",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "vgg19": "VGG19",
}


def run_one(cfg, name: str) -> dict:
    set_seed(int(cfg.seed))
    loaders = build_loaders(cfg)
    training = train_model(cfg, CHECKPOINTS / str(cfg.run_name), loaders)
    report, proba, labels = _train_script.evaluate_run(cfg, training["checkpoint"],
                                                       loaders)
    np.savez(RESULTS / f"predictions_{cfg.run_name}.npz", proba=proba, labels=labels)
    from osteognn.train import load_checkpoint
    model = load_checkpoint(cfg, training["checkpoint"])
    params = sum(p.numel() for p in model.parameters())
    save_json({"run": str(cfg.run_name), "config": dict(cfg), "training": training,
               "test": report, "parameters": int(params)},
              RESULTS / f"run_{cfg.run_name}.json")
    report["parameters"] = int(params)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--probe", action="store_true",
                        help="also run the augment-before-split protocol probe")
    parser.add_argument("--full-run", default="full")
    args = parser.parse_args()

    base = load_config(args.config, args.overrides)
    rows: dict[str, dict] = {}

    for name in (args.only or ORDER):
        if name not in BASELINES:
            raise SystemExit(f"unknown baseline {name!r}; known: {ORDER}")
        cfg = variant_config(base, name)
        print(f"\n=== baseline {name} ===", flush=True)
        report = run_one(cfg, name)
        rows[name] = {"label": LABELS[name], "test": report}
        print(f"    acc {report['accuracy']*100:.2f}%  "
              f"macroF1 {report['f1_macro']:.4f}  AUC {report['auc_macro_ovr']:.4f}  "
              f"params {report['parameters']/1e6:.2f}M", flush=True)

    # -- the proposed model, for the same table -----------------------------------
    full_path = RESULTS / f"run_{args.full_run}.json"
    if full_path.exists():
        full = load_json(full_path)
        rows["proposed"] = {"label": "EfficientNet-GCN (proposed)", "test": full["test"]}
        # Paired significance against every baseline on the identical test partition.
        proposed_pred = np.load(
            RESULTS / f"predictions_{args.full_run}.npz")["proba"].argmax(1)
        labels = np.load(RESULTS / f"predictions_{args.full_run}.npz")["labels"]
        for name in rows:
            if name == "proposed":
                continue
            path = RESULTS / f"predictions_{variant_config(base, name).run_name}.npz"
            if not path.exists():
                continue
            other = np.load(path)["proba"].argmax(1)
            rows[name]["mcnemar_vs_proposed"] = mcnemar_test(labels, proposed_pred, other)

    if args.probe:
        print("\n=== protocol probe: augment BEFORE split ===", flush=True)
        cfg = base.copy_with(run_name="probe_leaky_split",
                             data__processed_dir=str(base.data.processed_dir) + "_leaky")
        report = run_one(cfg, "leaky_split")
        rows["probe_leaky_split"] = {
            "label": "Proposed model, augment-before-split ordering",
            "test": report,
            "note": ("Not a baseline architecture. Identical network and budget; only "
                     "the ordering of augmentation and partitioning differs, so the "
                     "gap against the full model isolates the optimism that ordering "
                     "introduces."),
        }
        print(f"    acc {report['accuracy']*100:.2f}%  "
              f"macroF1 {report['f1_macro']:.4f}", flush=True)

    save_json(rows, RESULTS / "baselines.json")
    with open(RESULTS / "table_baselines.csv", "w", encoding="utf-8") as handle:
        handle.write("model,accuracy,macro_f1,auc,parameters_millions,mcnemar_p\n")
        for key, row in rows.items():
            t = row["test"]
            params = t.get("parameters")
            p = row.get("mcnemar_vs_proposed", {}).get("p_value", "")
            handle.write(f"\"{row['label']}\",{t['accuracy']:.4f},{t['f1_macro']:.4f},"
                         f"{t['auc_macro_ovr']:.4f},"
                         f"{'' if params is None else f'{params/1e6:.2f}'},"
                         f"{'' if p == '' else f'{p:.4f}'}\n")
    print(f"\nwrote {RESULTS/'table_baselines.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
