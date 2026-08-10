"""The leakage-control claims are the paper's main methodological contribution.

If any of these fail, the reported metrics are not what the manuscript says they are.

    pytest tests/test_protocol_integrity.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from osteognn.config import load_config
from osteognn.data.split import stratified_split, verify_no_leakage
from osteognn.metrics import (expected_calibration_error, mean_absolute_grade_error,
                              quadratic_weighted_kappa, top_k_accuracy,
                              wilson_interval)


def _fake_corpus():
    return {
        "Normal": [Path(f"Normal {i}.png") for i in range(780)],
        "Osteopenia": [Path(f"Osteopenia {i}.png") for i in range(374)],
        "Osteoporosis": [Path(f"Osteoporosis {i}.png") for i in range(793)],
    }


def test_split_reproduces_the_manuscript_table():
    split = stratified_split(_fake_corpus(), 0.70, 0.15, 0.15, seed=42)
    counts = {p: {c: sum(1 for _, lab in items if lab == c)
                  for c in ("Normal", "Osteopenia", "Osteoporosis")}
              for p, items in split.items()}
    assert counts["train"] == {"Normal": 546, "Osteopenia": 261, "Osteoporosis": 555}
    assert counts["val"] == {"Normal": 117, "Osteopenia": 56, "Osteoporosis": 119}
    assert counts["test"] == {"Normal": 117, "Osteopenia": 57, "Osteoporosis": 119}
    assert sum(counts["train"].values()) == 1362
    assert sum(counts["val"].values()) == 292
    assert sum(counts["test"].values()) == 293


def test_split_is_disjoint_and_total():
    split = stratified_split(_fake_corpus(), 0.70, 0.15, 0.15, seed=42)
    assert verify_no_leakage(split) == {"train_val": 0, "train_test": 0, "val_test": 0}
    everything = [p for items in split.values() for p, _ in items]
    assert len(everything) == len(set(everything)) == 1947


def test_leakage_check_catches_an_augmented_variant():
    """The stem comparison, not path equality, is what makes the check meaningful."""
    bad = {
        "train": [("data/train/Normal_5_aug1.png", "Normal")],
        "val": [],
        "test": [("data/test/Normal_5.png", "Normal")],
    }
    with pytest.raises(AssertionError, match="LEAKAGE"):
        verify_no_leakage(bad)


def test_seeds_produce_different_partitions():
    a = stratified_split(_fake_corpus(), 0.70, 0.15, 0.15, seed=42)
    b = stratified_split(_fake_corpus(), 0.70, 0.15, 0.15, seed=7)
    assert {p for p, _ in a["test"]} != {p for p, _ in b["test"]}


# ---------------------------------------------------------------------------------
def test_wilson_interval_against_known_values():
    lo, hi = wilson_interval(243, 293)
    assert 0.77 < lo < 0.79 and 0.86 < hi < 0.88
    # Degenerate cases must stay inside [0, 1] -- the reason Wilson is used here.
    assert wilson_interval(0, 57)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson_interval(57, 57)[1] == pytest.approx(1.0, abs=1e-12)


def test_quadratic_kappa_penalises_by_grade_distance():
    truth = np.array([0, 1, 2] * 20)
    adjacent = np.array([1, 0, 1] * 20)       # every error one grade away
    extreme = np.array([2, 1, 0] * 20)        # Normal <-> Osteoporosis
    assert quadratic_weighted_kappa(truth, adjacent) > \
           quadratic_weighted_kappa(truth, extreme)


def test_mean_absolute_grade_error():
    assert mean_absolute_grade_error(np.array([0, 1, 2]), np.array([0, 1, 2])) == 0.0
    assert mean_absolute_grade_error(np.array([0, 0]), np.array([2, 1])) == 1.5


def test_top2_accuracy_upper_bounds_accuracy():
    rng = np.random.RandomState(0)
    proba = rng.dirichlet(np.ones(3), size=200)
    labels = rng.randint(0, 3, 200)
    top1 = float((proba.argmax(1) == labels).mean())
    assert top_k_accuracy(labels, proba, 2) >= top1


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    labels = np.array([0] * 50 + [1] * 50)
    proba = np.zeros((100, 2))
    proba[:50] = [1.0, 0.0]
    proba[50:] = [0.0, 1.0]
    assert expected_calibration_error(labels, proba)["ece"] == pytest.approx(0.0, abs=1e-9)


def test_config_variants_change_exactly_one_thing():
    from osteognn.config import ABLATIONS, variant_config
    base = load_config(REPO_ROOT / "configs" / "default.yaml")
    for name in ABLATIONS:
        cfg = variant_config(base, name)
        differing = [k for k in ("model", "preprocess", "augment", "inference")
                     if json.dumps(cfg[k], sort_keys=True) !=
                     json.dumps(base[k], sort_keys=True)]
        assert len(differing) <= 1, f"{name} changed {differing}"
