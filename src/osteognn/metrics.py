"""Every quantity the manuscript reports, computed from labels and probabilities.

Precision, recall and F1 are macro averages -- each class weighted equally regardless of
support -- because the test partition is deliberately left at its natural, imbalanced
distribution. Two ordinal measures are reported alongside them, since the severity
grades are ordered rather than merely categorical: a Normal/Osteoporosis confusion is
clinically far costlier than an adjacent-grade one, and nominal accuracy treats the two
identically.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score,
                             roc_curve)


# ---------------------------------------------------------------------------------
# Interval estimates
# ---------------------------------------------------------------------------------
def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval -- preferred to the normal approximation at these sizes.

    The Osteopenia partition holds 57 images; the normal approximation is unreliable
    there and can even produce bounds outside [0, 1].
    """
    if total == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_interval(y_true: np.ndarray, y_pred: np.ndarray, metric, n: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over test images, for metrics with no closed-form interval."""
    rng = np.random.RandomState(seed)
    size = len(y_true)
    values = []
    for _ in range(n):
        idx = rng.randint(0, size, size)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            values.append(metric(y_true[idx], y_pred[idx]))
        except ValueError:
            continue
    if not values:
        return (float("nan"), float("nan"))
    return (float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------------------
# Ordinal measures
# ---------------------------------------------------------------------------------
def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray,
                             n_classes: int = 3) -> float:
    """kappa_w with w_ij = (i-j)^2 / (C-1)^2 (Eq. 9)."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                   labels=list(range(n_classes))))


def mean_absolute_grade_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean |g_hat - g| over grades encoded 0, 1, 2."""
    return float(np.mean(np.abs(y_pred.astype(int) - y_true.astype(int))))


def adjacent_and_extreme_errors(cm: np.ndarray) -> dict[str, int]:
    """Split the error mass into adjacent-grade and Normal<->Osteoporosis confusions."""
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    total = int(errors.sum())
    extreme = int(errors[0, 2] + errors[2, 0])
    return {"errors": total, "adjacent": total - extreme, "extreme": extreme,
            "correct": int(np.trace(cm))}


# ---------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------
def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                               n_bins: int = 15) -> dict:
    """ECE over equal-width confidence bins, plus the reliability-diagram data."""
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "accuracy": None, "confidence": None})
            continue
        acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
        ece += mask.mean() * abs(acc - conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                     "accuracy": acc, "confidence": conf})
    return {"ece": float(ece), "bins": bins,
            "mean_confidence": float(confidence.mean()),
            "accuracy": float(correct.mean())}


# ---------------------------------------------------------------------------------
# Top-k
# ---------------------------------------------------------------------------------
def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int = 2) -> float:
    """Fraction of images whose true grade is among the k highest-ranked classes.

    With three classes top-2 is a weak upper bound by construction, but it is precisely
    the quantity that tests whether a high AUC alongside a lower arg-max accuracy means
    the model ranks sensibly where it decides wrongly.
    """
    order = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, order)]))


# ---------------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------------
def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, classes: list[str],
                         ece_bins: int = 15, alpha: float = 0.05,
                         bootstrap_n: int = 2000, seed: int = 0) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=np.float64)
    y_pred = proba.argmax(axis=1)
    n_classes = len(classes)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    onehot = np.eye(n_classes)[y_true]
    per_class_auc, roc_points = {}, {}
    for i, name in enumerate(classes):
        if len(np.unique(onehot[:, i])) < 2:
            per_class_auc[name] = float("nan")
            continue
        per_class_auc[name] = float(roc_auc_score(onehot[:, i], proba[:, i]))
        fpr, tpr, _ = roc_curve(onehot[:, i], proba[:, i])
        roc_points[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    macro_auc = float(roc_auc_score(onehot, proba, average="macro", multi_class="ovr"))

    correct = int((y_pred == y_true).sum())
    acc_ci = wilson_interval(correct, len(y_true), alpha)
    per_class = {}
    for i, name in enumerate(classes):
        hits = int(cm[i, i])
        total = int(cm[i].sum())
        lo, hi = wilson_interval(hits, total, alpha)
        per_class[name] = {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
            "auc": per_class_auc[name],
            "recall_ci": [lo, hi],
        }

    f1_lo, f1_hi = bootstrap_interval(
        y_true, y_pred,
        lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        n=bootstrap_n, alpha=alpha, seed=seed)

    return {
        "n": int(len(y_true)),
        "accuracy": accuracy,
        "accuracy_ci": list(acc_ci),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro_ci": [f1_lo, f1_hi],
        "auc_macro_ovr": macro_auc,
        "auc_per_class": per_class_auc,
        "top2_accuracy": top_k_accuracy(y_true, proba, 2),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, n_classes),
        "mae_grade": mean_absolute_grade_error(y_true, y_pred),
        "confusion_matrix": cm.tolist(),
        "error_structure": adjacent_and_extreme_errors(cm),
        "calibration": expected_calibration_error(y_true, proba, ece_bins),
        "per_class": per_class,
        "roc": roc_points,
        "classes": list(classes),
    }


# ---------------------------------------------------------------------------------
# Paired comparison against a baseline
# ---------------------------------------------------------------------------------
def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on the discordant pairs of two classifiers, same test set."""
    a_right, b_right = pred_a == y_true, pred_b == y_true
    b01 = int(np.sum(a_right & ~b_right))    # a correct, b wrong
    b10 = int(np.sum(~a_right & b_right))    # b correct, a wrong
    n = b01 + b10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b01, b10), n, 0.5)))
    return {"a_only_correct": b01, "b_only_correct": b10, "n_discordant": n,
            "p_value": p}
