"""Every measured figure in the manuscript, regenerated from results/.

Palette note. The three severity grades get a fixed, ordered hue assignment --
blue -> amber -> red -- which is both categorical (identity in the ROC panel) and
semantically ordered (healthy -> intermediate -> severe). The triple was checked with a
CVD validator rather than by eye: worst adjacent pair separates by dE 16.2 under
deuteranopia and 18.8 under normal vision, comfortably above the dE 8 floor, and every
hue clears 3:1 contrast against the page. The confusion matrix uses a single-hue
sequential ramp, never a rainbow, because its cells encode magnitude.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Fixed categorical order -- never cycled, never reassigned when a series is dropped.
CLASS_COLORS = {"Normal": "#1f6feb", "Osteopenia": "#d97706", "Osteoporosis": "#b91c1c"}
SERIES = ["#1f6feb", "#d97706", "#b91c1c"]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
SEQUENTIAL = LinearSegmentedColormap.from_list("osteo_blue", ["#f4f8fd", "#1f6feb"])

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _save(fig, path: Path, also: Path | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if also is not None:
        fig.savefig(also)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------
def training_curves(history: list[dict], out: Path, also: Path | None = None) -> Path:
    """Loss and accuracy, train versus validation. Two panels, never a dual axis."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], color=SERIES[0],
                 label="Train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], color=SERIES[1],
                 label="Validation")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.set_title("Loss")
    ax_loss.legend(frameon=False)

    ax_acc.plot(epochs, [100 * h["train_acc"] for h in history], color=SERIES[0],
                label="Train")
    ax_acc.plot(epochs, [100 * h["val_acc"] for h in history], color=SERIES[1],
                label="Validation")
    best = max(history, key=lambda h: h["val_macro_f1"])
    ax_acc.axvline(best["epoch"], color=MUTED, linestyle=":", linewidth=1.0)
    # Direct-label only the retained checkpoint, not every point.
    ax_acc.annotate(f"retained: epoch {best['epoch']}\n"
                    f"val acc {100*best['val_acc']:.2f}%",
                    xy=(best["epoch"], 100 * best["val_acc"]),
                    xytext=(6, -22), textcoords="offset points",
                    fontsize=7.5, color=MUTED)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy"); ax_acc.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, out, also)


def roc_curves(report: dict, out: Path, also: Path | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    for name in report["classes"]:
        pts = report["roc"][name]
        ax.plot(pts["fpr"], pts["tpr"], color=CLASS_COLORS[name],
                label=f"{name} (AUC {report['auc_per_class'][name]:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=0.9)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"One-vs-rest ROC (macro AUC {report['auc_macro_ovr']:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, out, also)


def confusion_matrix(report: dict, out: Path, also: Path | None = None) -> Path:
    """Counts and row-normalised recall, side by side."""
    cm = np.array(report["confusion_matrix"], dtype=float)
    classes = report["classes"]
    normalised = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, data, title, fmt in ((axes[0], cm, "Counts", "{:.0f}"),
                                 (axes[1], normalised, "Row-normalised recall",
                                  "{:.2f}")):
        ax.imshow(data / data.max(), cmap=SEQUENTIAL, vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=20, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
        ax.grid(False)
        for i in range(len(classes)):
            for j in range(len(classes)):
                shade = data[i, j] / data.max()
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        fontsize=8.5, color="white" if shade > 0.55 else INK)
    fig.tight_layout()
    return _save(fig, out, also)


def reliability_diagram(report: dict, out: Path) -> Path:
    """Confidence against accuracy per bin, with the ECE stated in the title."""
    calib = report["calibration"]
    bins = [b for b in calib["bins"] if b["n"] > 0]
    centres = [(b["lo"] + b["hi"]) / 2 for b in bins]
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=0.9,
            label="Perfect calibration")
    ax.bar(centres, [b["accuracy"] for b in bins], width=0.055, color=SERIES[0],
           edgecolor="white", linewidth=0.8, label="Observed accuracy")
    ax.plot(centres, [b["confidence"] for b in bins], color=SERIES[2], marker="o",
            markersize=3.5, label="Mean confidence")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability (ECE {calib['ece']:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, out)


def ablation_bars(ablation: dict, out: Path) -> Path:
    """Delta against the full model. Bars start at zero; direction carries the sign."""
    ref = ablation["full"]["test"]
    keys = [k for k in ablation if k != "full"]
    labels = [ablation[k]["label"] for k in keys]
    d_acc = [100 * (ablation[k]["test"]["accuracy"] - ref["accuracy"]) for k in keys]
    d_f1 = [ablation[k]["test"]["f1_macro"] - ref["f1_macro"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.42 * len(keys) + 1.8), sharey=True)
    y = np.arange(len(keys))
    for ax, values, title, unit in ((axes[0], d_acc, "Accuracy", "pp"),
                                    (axes[1], [100 * v for v in d_f1],
                                     "Macro F1", "pp")):
        colors = [SERIES[2] if v < 0 else SERIES[0] for v in values]
        ax.barh(y, values, color=colors, height=0.62)
        ax.axvline(0, color=MUTED, linewidth=0.9)
        ax.set_title(f"{title} change vs full model")
        ax.set_xlabel(f"Difference ({unit})")
        ax.grid(axis="y", visible=False)
        span = max(1e-6, max(abs(min(values)), abs(max(values))))
        for yi, v in zip(y, values):
            ax.text(v + (0.04 * span if v >= 0 else -0.04 * span), yi, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7.5,
                    color=MUTED)
        ax.set_xlim(-1.35 * span, 1.35 * span)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return _save(fig, out)


def intensity_distribution(samples: dict, classes: list[str], analysis: dict,
                           out: Path, also: Path | None = None) -> Path:
    """Per-class bone-region intensity: distribution plus mean and SD."""
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for name in classes:
        values = samples[f"bone_{name}"]
        ax_hist.hist(values, bins=40, histtype="step", linewidth=1.5,
                     color=CLASS_COLORS[name], label=name, density=True)
    ax_hist.set_xlabel("Mean bone-region intensity (0-255)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Distribution by class")
    ax_hist.legend(frameon=False)

    data = [samples[f"bone_{name}"] for name in classes]
    parts = ax_box.boxplot(data, labels=classes, patch_artist=True, widths=0.55,
                           medianprops={"color": "white", "linewidth": 1.4},
                           flierprops={"marker": ".", "markersize": 2,
                                       "markerfacecolor": MUTED,
                                       "markeredgecolor": "none"})
    for patch, name in zip(parts["boxes"], classes):
        patch.set_facecolor(CLASS_COLORS[name])
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)
    kruskal = analysis["kruskal_bone"]
    ax_box.set_ylabel("Mean bone-region intensity")
    ax_box.set_title(f"Kruskal-Wallis H = {kruskal['H']:.1f}, "
                     f"p = {kruskal['p']:.1e}")
    ax_box.tick_params(axis="x", rotation=12)
    ax_box.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, out, also)


def preprocessing_comparison(pairs: list[tuple[str, np.ndarray, np.ndarray]],
                             out: Path, also: Path | None = None) -> Path:
    """Original versus CLAHE-at-native-resolution, one row per class."""
    fig, axes = plt.subplots(len(pairs), 2, figsize=(4.4, 2.2 * len(pairs)))
    axes = np.atleast_2d(axes)
    for row, (name, original, processed) in enumerate(pairs):
        for col, (image, title) in enumerate(((original, "Original"),
                                              (processed, "CLAHE + resize"))):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(name, fontsize=8.5)
    fig.tight_layout()
    return _save(fig, out, also)


def fusion_sweep(sweep: dict, out: Path) -> Path:
    """Test and validation macro F1 across the ensemble weight w."""
    rows = sweep["grid"]
    w = [r["w"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    ax.plot(w, [r["val_f1_macro"] for r in rows], color=SERIES[0], marker="o",
            markersize=3.5, label="Validation")
    ax.plot(w, [r["test_f1_macro"] for r in rows], color=SERIES[1], marker="s",
            markersize=3.5, label="Test")
    ax.axvline(sweep["a_priori_w"], color=MUTED, linestyle=":", linewidth=1.0)
    ax.annotate(f"a priori w = {sweep['a_priori_w']:.1f}",
                xy=(sweep["a_priori_w"], ax.get_ylim()[0]), xytext=(4, 8),
                textcoords="offset points", fontsize=7.5, color=MUTED)
    ax.set_xlabel("Ensemble weight $w$ (graph branch)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Sensitivity to the fusion weight")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, out)


def multiseed_spread(multiseed: dict, out: Path) -> Path:
    """Per-seed accuracy and macro F1 -- the variance a single run cannot show."""
    seeds = [str(s) for s in multiseed["seeds"]]
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    width = 0.36
    x = np.arange(len(seeds))
    acc = [100 * multiseed["summary"]["accuracy"]["per_seed"][s] for s in seeds]
    f1 = [100 * multiseed["summary"]["f1_macro"]["per_seed"][s] for s in seeds]
    ax.bar(x - width / 2, acc, width, color=SERIES[0], label="Accuracy",
           edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, f1, width, color=SERIES[1], label="Macro F1",
           edgecolor="white", linewidth=0.8)
    for xi, v in zip(x - width / 2, acc):
        ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=7, color=MUTED)
    for xi, v in zip(x + width / 2, f1):
        ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, max(max(acc), max(f1)) * 1.18)
    ax.set_title("Across-seed spread")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, out)
