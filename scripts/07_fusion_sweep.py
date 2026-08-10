"""Sensitivity of the fused prediction to the ensemble weight w.

w = 0.6 was fixed a priori rather than tuned. This sweep says what that choice cost.
The sweep is run on the *validation* partition -- selecting w on the test partition
would be exactly the kind of optimism the rest of the protocol exists to avoid -- and
the test number is then reported at both the a-priori w and the validation-optimal w.

    python scripts/07_fusion_sweep.py
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import CHECKPOINTS, REPO_ROOT, RESULTS

from osteognn.config import load_config
from osteognn.data.datasets import build_loaders
from osteognn.metrics import evaluate_predictions
from osteognn.train import collect_probabilities, load_checkpoint
from osteognn.utils import get_device, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run", default="full")
    parser.add_argument("--grid", nargs="+", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    model = load_checkpoint(cfg, CHECKPOINTS / args.run / "checkpoint.pt", device)
    loaders = build_loaders(cfg, partitions=("val", "test"))
    classes = list(cfg.data.classes)
    tta = bool(cfg.inference.tta_hflip)

    rows = []
    for w in args.grid:
        val_p, val_y = collect_probabilities(model, loaders["val"], device, tta, w)
        test_p, test_y = collect_probabilities(model, loaders["test"], device, tta, w)
        val = evaluate_predictions(val_y, val_p, classes, bootstrap_n=0)
        test = evaluate_predictions(test_y, test_p, classes, bootstrap_n=0)
        rows.append({"w": w,
                     "val_accuracy": val["accuracy"], "val_f1_macro": val["f1_macro"],
                     "test_accuracy": test["accuracy"], "test_f1_macro": test["f1_macro"],
                     "test_auc": test["auc_macro_ovr"]})
        print(f"w={w:.1f}  val acc {val['accuracy']*100:5.2f}% f1 {val['f1_macro']:.4f} "
              f"| test acc {test['accuracy']*100:5.2f}% f1 {test['f1_macro']:.4f}",
              flush=True)

    best = max(rows, key=lambda r: r["val_f1_macro"])
    apriori = min(rows, key=lambda r: abs(r["w"] - float(cfg.inference.fusion_weight)))
    summary = {
        "grid": rows,
        "selected_on": "validation macro F1",
        "best_w": best["w"], "best": best,
        "a_priori_w": apriori["w"], "a_priori": apriori,
        "test_f1_delta": best["test_f1_macro"] - apriori["test_f1_macro"],
        "test_accuracy_delta": best["test_accuracy"] - apriori["test_accuracy"],
        "test_accuracy_range": [min(r["test_accuracy"] for r in rows),
                                max(r["test_accuracy"] for r in rows)],
    }
    print(f"\nvalidation-optimal w = {best['w']:.1f}; a priori w = {apriori['w']:.1f}; "
          f"test macro F1 differs by {summary['test_f1_delta']:+.4f}")

    save_json(summary, RESULTS / "fusion_sweep.json")
    with open(RESULTS / "table_fusion_sweep.csv", "w", encoding="utf-8") as handle:
        handle.write("w,val_accuracy,val_macro_f1,test_accuracy,test_macro_f1,test_auc\n")
        for r in rows:
            handle.write(f"{r['w']:.1f},{r['val_accuracy']:.4f},{r['val_f1_macro']:.4f},"
                         f"{r['test_accuracy']:.4f},{r['test_f1_macro']:.4f},"
                         f"{r['test_auc']:.4f}\n")
    print(f"wrote {RESULTS/'table_fusion_sweep.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
