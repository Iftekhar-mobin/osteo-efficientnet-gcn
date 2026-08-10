"""Repeat the full protocol across seeds and report the variance a single run cannot.

Wilson intervals capture sampling uncertainty within one fixed test partition. They say
nothing about how much the number would move if the network were trained again, which is
a separate and often larger source of variation. Each seed re-draws the partition as
well as the initialisation, so the spread reported here covers both.

    python scripts/05_multiseed.py --seeds 42 7 1234
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT, RESULTS

from osteognn.config import load_config
from osteognn.utils import load_json, save_json

METRICS = ["accuracy", "f1_macro", "auc_macro_ovr", "precision_macro", "recall_macro",
           "qwk", "mae_grade", "top2_accuracy"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 1234])
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs = {}
    for seed in args.seeds:
        run_name = f"seed{seed}"
        result_path = RESULTS / f"run_{run_name}.json"
        if not (args.skip_existing and result_path.exists()):
            # Each seed re-splits: a fixed partition with varying initialisation would
            # under-report the variance that matters, since partition choice on 293 test
            # images is itself a large source of movement.
            processed = f"{cfg.data.processed_dir}_seed{seed}"
            prep = [sys.executable, str(REPO_ROOT / "scripts" / "01_preprocess_and_split.py"),
                    "--set", f"seed={seed}", "--set", f"data.processed_dir={processed}",
                    "--skip-intensity"]
            if args.raw_dir:
                prep += ["--raw-dir", args.raw_dir]
            print(f"\n=== seed {seed}: preprocessing ===", flush=True)
            subprocess.run(prep, check=True)
            print(f"=== seed {seed}: training ===", flush=True)
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "02_train.py"),
                 "--set", f"seed={seed}", "--set", f"run_name={run_name}",
                 "--set", f"data.processed_dir={processed}"], check=True)
        runs[seed] = load_json(result_path)["test"]

    summary = {}
    for metric in METRICS:
        values = np.array([runs[s][metric] for s in args.seeds], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()),
            "per_seed": {str(s): float(runs[s][metric]) for s in args.seeds},
        }
        print(f"{metric:18s} {summary[metric]['mean']:.4f} "
              f"+/- {summary[metric]['sd']:.4f}  "
              f"[{summary[metric]['min']:.4f}, {summary[metric]['max']:.4f}]")

    save_json({"seeds": args.seeds, "summary": summary}, RESULTS / "multiseed.json")
    with open(RESULTS / "table_multiseed.csv", "w", encoding="utf-8") as handle:
        handle.write("metric,mean,sd,min,max," + ",".join(f"seed{s}" for s in args.seeds) + "\n")
        for metric in METRICS:
            s = summary[metric]
            per = ",".join(f"{s['per_seed'][str(seed)]:.4f}" for seed in args.seeds)
            handle.write(f"{metric},{s['mean']:.4f},{s['sd']:.4f},{s['min']:.4f},"
                         f"{s['max']:.4f},{per}\n")
    print(f"\nwrote {RESULTS/'table_multiseed.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
