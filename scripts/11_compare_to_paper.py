"""Diff a fresh run against the manuscript's originally reported values.

Exits non-zero when a headline metric moves by more than the tolerance, so a divergence
is visible rather than silent. A non-zero exit is not a failure of the code -- it means
this run does not reproduce the pre-repository numbers, which is information, not a bug.

    python scripts/11_compare_to_paper.py
    python scripts/11_compare_to_paper.py --tolerance 0.02
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS

from osteognn.utils import load_json, save_json

HEADLINE = [
    ("accuracy", "Accuracy", "test.accuracy"),
    ("precision_macro", "Precision (macro)", "test.precision_macro"),
    ("recall_macro", "Recall (macro)", "test.recall_macro"),
    ("f1_macro", "F1 (macro)", "test.f1_macro"),
    ("auc_macro_ovr", "AUC (macro OvR)", "test.auc_macro_ovr"),
]


def dig(tree, dotted):
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="full")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="absolute difference tolerated per metric")
    args = parser.parse_args()

    reported = load_json(RESULTS / "paper_reported.json")
    run = load_json(RESULTS / f"run_{args.run}.json")
    measured = {"test": run["test"]}

    rows, breaches = [], []
    print(f"{'metric':22s} {'reported':>10s} {'measured':>10s} {'delta':>9s}")
    print("-" * 55)
    for key, label, path in HEADLINE:
        want, got = dig(reported, path), dig(measured, path)
        if want is None or got is None:
            continue
        delta = got - want
        flag = "" if abs(delta) <= args.tolerance else "  BREACH"
        if flag:
            breaches.append(label)
        rows.append({"metric": label, "reported": want, "measured": got,
                     "delta": delta, "within_tolerance": not flag})
        print(f"{label:22s} {want:10.4f} {got:10.4f} {delta:+9.4f}{flag}")

    # Structural quantities that must match exactly, not within a tolerance.
    print()
    exact = []
    eff_path = RESULTS / "efficiency.json"
    if eff_path.exists():
        eff = load_json(eff_path)
        for label, want, got in (
            ("total parameters", reported["model"]["total_parameters"],
             eff["total_parameters"]),
            ("trainable parameters", reported["model"]["trainable_parameters"],
             eff["trainable_parameters"]),
        ):
            ok = want == got
            exact.append({"quantity": label, "reported": want, "measured": got,
                          "exact_match": ok})
            print(f"{label:22s} {want:10,d} {got:10,d}   {'MATCH' if ok else 'DIFFERS'}")

    data_path = RESULTS / "dataset_summary.json"
    if data_path.exists():
        data = load_json(data_path)
        for part in ("train", "val", "test"):
            want = reported["dataset"][{"train": "train_original"}.get(part, part)]
            got = data["totals"][part]
            ok = want == got
            exact.append({"quantity": f"{part} partition size", "reported": want,
                          "measured": got, "exact_match": ok})
            print(f"{part + ' partition':22s} {want:10,d} {got:10,d}   "
                  f"{'MATCH' if ok else 'DIFFERS'}")

    save_json({"tolerance": args.tolerance, "headline": rows, "structural": exact,
               "breaches": breaches}, RESULTS / "compare_to_paper.json")
    with open(RESULTS / "compare_to_paper.csv", "w", encoding="utf-8") as handle:
        handle.write("metric,reported,measured,delta,within_tolerance\n")
        for row in rows:
            handle.write(f"\"{row['metric']}\",{row['reported']:.4f},"
                         f"{row['measured']:.4f},{row['delta']:+.4f},"
                         f"{row['within_tolerance']}\n")

    if breaches:
        print(f"\n{len(breaches)} metric(s) outside +/-{args.tolerance}: "
              f"{', '.join(breaches)}")
        print("This run does not reproduce the pre-repository numbers. The manuscript "
              "reports the measured values; see results/run_*.json.")
        return 1
    print(f"\nall headline metrics within +/-{args.tolerance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
