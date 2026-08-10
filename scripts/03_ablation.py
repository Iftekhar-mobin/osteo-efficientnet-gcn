"""Run the seven-variant ablation protocol.

Exactly one component is removed or substituted at a time; the data splits, backbone and
training budget are held fixed for every variant. Two of the seven are inference-time
changes to the trained full model and require no retraining -- (ii) reads the graph
branch's softmax alone, (vi) drops the horizontal-flip averaging -- so they are computed
from the full model's checkpoint rather than from a fresh run.

    python scripts/03_ablation.py                    # all variants
    python scripts/03_ablation.py --only conv_control
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import CHECKPOINTS, REPO_ROOT, RESULTS

from osteognn.config import ABLATIONS, is_inference_only, load_config, variant_config
from osteognn.data.datasets import build_loaders
from osteognn.utils import load_json, save_json, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_train_script = import_module("02_train")

ORDER = ["no_gcn", "no_aux", "mean_pool", "no_clahe", "no_balance", "no_tta",
         "conv_control"]
LABELS = {
    "no_gcn": "(i) w/o GCN branch",
    "no_aux": "(ii) w/o auxiliary branch",
    "mean_pool": "(iii) w/o dual mean-max pooling",
    "no_clahe": "(iv) w/o CLAHE preprocessing",
    "no_balance": "(v) w/o class-balancing augmentation",
    "no_tta": "(vi) w/o test-time augmentation",
    "conv_control": "(vii) GCN -> param-matched conv",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--full-run", default="full",
                        help="run name of the trained full model")
    args = parser.parse_args()

    base = load_config(args.config, args.overrides)
    full_path = RESULTS / f"run_{args.full_run}.json"
    if not full_path.exists():
        raise SystemExit(f"train the full model first: missing {full_path}")
    full = load_json(full_path)

    rows = {"full": {"label": "Full model (proposed)", "test": full["test"]}}
    wanted = args.only or ORDER

    for name in wanted:
        if name not in ABLATIONS:
            raise SystemExit(f"unknown ablation {name!r}; known: {ORDER}")
        cfg = variant_config(base, name)
        print(f"\n=== ablation {name}: {LABELS[name]} ===", flush=True)

        if is_inference_only(name):
            # Re-evaluate the retained full-model checkpoint under the altered
            # inference rule. No training budget is consumed and no weights change,
            # which is what makes these two rows exact rather than approximate.
            eval_cfg = base.copy_with(**{
                "inference__fusion_weight": float(cfg.inference.fusion_weight),
                "inference__tta_hflip": bool(cfg.inference.tta_hflip)})
            loaders = build_loaders(eval_cfg, partitions=("test",))
            checkpoint = CHECKPOINTS / args.full_run / "checkpoint.pt"
            report, proba, labels = _train_script.evaluate_run(
                eval_cfg, checkpoint, loaders,
                tta=bool(cfg.inference.tta_hflip),
                fusion_weight=float(cfg.inference.fusion_weight))
            np.savez(RESULTS / f"predictions_{cfg.run_name}.npz",
                     proba=proba, labels=labels)
            save_json({"run": str(cfg.run_name), "inference_only": True,
                       "test": report}, RESULTS / f"run_{cfg.run_name}.json")
        else:
            if not bool(cfg.preprocess.clahe):
                cfg.set_path("data.processed_dir",
                             str(base.data.processed_dir) + "_noclahe")
            if not bool(cfg.augment.balance):
                # ablation (v) trains on the raw distribution: the unbalanced training
                # index is written by scripts/01 with augment.balance=false
                cfg.set_path("data.processed_dir",
                             str(base.data.processed_dir) + "_unbalanced")
            set_seed(int(cfg.seed))
            loaders = build_loaders(cfg)
            from osteognn.train import train_model
            training = train_model(cfg, CHECKPOINTS / str(cfg.run_name), loaders)
            report, proba, labels = _train_script.evaluate_run(
                cfg, training["checkpoint"], loaders)
            np.savez(RESULTS / f"predictions_{cfg.run_name}.npz",
                     proba=proba, labels=labels)
            save_json({"run": str(cfg.run_name), "config": dict(cfg),
                       "training": training, "test": report},
                      RESULTS / f"run_{cfg.run_name}.json")

        rows[name] = {"label": LABELS[name], "test": report,
                      "inference_only": is_inference_only(name)}
        print(f"    acc {report['accuracy']*100:.2f}%  "
              f"macroF1 {report['f1_macro']:.4f}  AUC {report['auc_macro_ovr']:.4f}",
              flush=True)

    save_json(rows, RESULTS / "ablation.json")
    with open(RESULTS / "table_ablation.csv", "w", encoding="utf-8") as handle:
        handle.write("variant,accuracy,macro_f1,auc,delta_accuracy,delta_macro_f1\n")
        ref = rows["full"]["test"]
        for key in ["full"] + [k for k in ORDER if k in rows]:
            t = rows[key]["test"]
            handle.write(f"\"{rows[key]['label']}\",{t['accuracy']:.4f},"
                         f"{t['f1_macro']:.4f},{t['auc_macro_ovr']:.4f},"
                         f"{t['accuracy']-ref['accuracy']:+.4f},"
                         f"{t['f1_macro']-ref['f1_macro']:+.4f}\n")
    print(f"\nwrote {RESULTS/'table_ablation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
