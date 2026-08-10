"""Train one configuration and evaluate it on the held-out test partition.

    python scripts/02_train.py                       # the full proposed model
    python scripts/02_train.py --variant no_gcn      # an ablation
    python scripts/02_train.py --set seed=7 --set run_name=seed7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import CHECKPOINTS, REPO_ROOT, RESULTS

from osteognn.config import load_config, variant_config
from osteognn.data.datasets import build_loaders
from osteognn.metrics import evaluate_predictions
from osteognn.train import collect_probabilities, load_checkpoint, train_model
from osteognn.utils import device_report, get_device, git_commit, save_json, set_seed

import numpy as np


def evaluate_run(cfg, checkpoint: str, loaders, tta: bool | None = None,
                 fusion_weight: float | None = None) -> dict:
    device = get_device()
    model = load_checkpoint(cfg, checkpoint, device)
    tta = bool(cfg.inference.tta_hflip) if tta is None else tta
    proba, labels = collect_probabilities(model, loaders["test"], device, tta,
                                          fusion_weight)
    report = evaluate_predictions(labels, proba, list(cfg.data.classes),
                                  ece_bins=int(cfg.eval.ece_bins),
                                  alpha=float(cfg.eval.wilson_alpha),
                                  bootstrap_n=int(cfg.eval.bootstrap_n),
                                  seed=int(cfg.seed))
    report["tta"] = tta
    report["fusion_weight"] = (float(cfg.inference.fusion_weight)
                               if fusion_weight is None else float(fusion_weight))
    return report, proba, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="2 epochs, for exercising the code path")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.variant:
        cfg = variant_config(cfg, args.variant)
    if args.quick:
        cfg.set_path("train.epochs", 2)
        cfg.set_path("train.warmup_epochs", 1)
        cfg.set_path("run_name", str(cfg.run_name) + "_quick")

    run_name = str(cfg.run_name)
    out_dir = CHECKPOINTS / run_name
    set_seed(int(cfg.seed))
    print(f"=== run {run_name} (seed {cfg.seed}) ===", flush=True)
    print(f"    device: {device_report()}", flush=True)

    loaders = build_loaders(cfg)
    print(f"    train {len(loaders['train'].dataset)} | "
          f"val {len(loaders['val'].dataset)} | "
          f"test {len(loaders['test'].dataset)}", flush=True)

    training = train_model(cfg, out_dir, loaders)
    print(f"    best epoch {training['best']['epoch']} "
          f"val macro F1 {training['best']['val_macro_f1']:.4f}", flush=True)

    report, proba, labels = evaluate_run(cfg, training["checkpoint"], loaders)
    print(f"    TEST acc {report['accuracy']*100:.2f}% "
          f"(95% CI {report['accuracy_ci'][0]*100:.2f}-{report['accuracy_ci'][1]*100:.2f}) "
          f"macroF1 {report['f1_macro']:.4f} AUC {report['auc_macro_ovr']:.4f}",
          flush=True)

    np.savez(RESULTS / f"predictions_{run_name}.npz", proba=proba, labels=labels)
    save_json({"run": run_name, "config": dict(cfg), "training": training,
               "test": report, "environment": device_report(),
               "git_commit": git_commit()},
              RESULTS / f"run_{run_name}.json")
    print(f"    wrote {RESULTS / f'run_{run_name}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
