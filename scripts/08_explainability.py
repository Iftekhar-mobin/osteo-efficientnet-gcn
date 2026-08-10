"""Grad-CAM, GNNExplainer and LIME, plus the deletion/insertion faithfulness measures.

Explanations are produced for both correctly classified and misclassified test images.
The failure cases are the point: a suite that only ever shows plausible maps for correct
predictions cannot exclude the possibility that it would produce equally plausible maps
for wrong ones, so plausibility on correct cases alone is not evidence.

    python scripts/08_explainability.py --run full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _bootstrap  # noqa: F401
from _bootstrap import CHECKPOINTS, FIGURES, REPO_ROOT, RESULTS

from osteognn.config import load_config
from osteognn.data.datasets import RadiographDataset, load_partition
from osteognn.figures import CLASS_COLORS, INK, MUTED, SEQUENTIAL, _save
from osteognn.train import load_checkpoint
from osteognn.utils import get_device, save_json, set_seed
from osteognn.xai.faithfulness import deletion_insertion, random_control
from osteognn.xai.gnn_explainer import explain_nodes
from osteognn.xai.gradcam import GradCAM, resolve_layer


def denormalise(tensor: torch.Tensor, cfg) -> np.ndarray:
    mean = np.array(cfg.preprocess.norm_mean).reshape(3, 1, 1)
    std = np.array(cfg.preprocess.norm_std).reshape(3, 1, 1)
    image = tensor.detach().cpu().numpy()[0] * std + mean
    return np.clip(image.transpose(1, 2, 0), 0, 1)


def pick_examples(records, labels, preds, classes, n_correct=3, n_wrong=2):
    """A few correct predictions spread across classes, plus the worst failures."""
    chosen = []
    for class_index, name in enumerate(classes):
        hits = np.where((labels == class_index) & (preds == labels))[0]
        if len(hits):
            chosen.append((int(hits[0]), name, "correct"))
        if len(chosen) >= n_correct:
            break
    extreme = np.where(np.abs(preds - labels) == 2)[0]      # Normal <-> Osteoporosis
    adjacent = np.where(np.abs(preds - labels) == 1)[0]
    for pool, kind in ((extreme, "extreme error"), (adjacent, "adjacent error")):
        for idx in pool[:1]:
            chosen.append((int(idx), classes[labels[idx]], kind))
        if len([c for c in chosen if "error" in c[2]]) >= n_wrong:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run", default="full")
    parser.add_argument("--skip-lime", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = get_device()
    classes = list(cfg.data.classes)
    model = load_checkpoint(cfg, CHECKPOINTS / args.run / "checkpoint.pt", device)

    records = load_partition(cfg.data.processed_dir, "test")
    dataset = RadiographDataset(records, classes, cfg, train=False)
    stored = np.load(RESULTS / f"predictions_{args.run}.npz")
    labels, preds = stored["labels"], stored["proba"].argmax(1)

    examples = pick_examples(records, labels, preds, classes)
    print(f"explaining {len(examples)} test images: "
          f"{[(i, k) for i, _, k in examples]}", flush=True)

    target_layer = resolve_layer(model, str(cfg.xai.gradcam_layer))
    summary: dict = {"examples": [], "faithfulness": {}}
    cams, gnn_masks, panels = [], [], []

    for index, true_name, kind in examples:
        image, _ = dataset[index]
        image = image.unsqueeze(0).to(device)
        predicted = int(preds[index])

        cam_tool = GradCAM(model, target_layer, branch="graph")
        cam, _ = cam_tool(image, predicted)
        cam_tool.remove()

        gnn = explain_nodes(model, image, predicted,
                            epochs=int(cfg.xai.gnnexplainer_epochs),
                            lr=float(cfg.xai.gnnexplainer_lr), seed=int(cfg.seed))

        # Faithfulness, against the random-order null that makes the AUCs readable.
        grid = gnn["node_mask_normalised"]
        upsampled = np.kron(grid, np.ones((image.shape[-2] // grid.shape[0],
                                           image.shape[-1] // grid.shape[1])))
        fa_cam = deletion_insertion(model, image, cam, predicted,
                                    int(cfg.xai.faithfulness_steps))
        fa_gnn = deletion_insertion(model, image, upsampled, predicted,
                                    int(cfg.xai.faithfulness_steps))
        fa_rnd = random_control(model, image, predicted,
                                int(cfg.xai.faithfulness_steps), seed=int(cfg.seed))

        summary["examples"].append({
            "index": index, "true": true_name, "predicted": classes[predicted],
            "kind": kind,
            "gradcam": {"deletion_auc": fa_cam["deletion_auc"],
                        "insertion_auc": fa_cam["insertion_auc"]},
            "gnnexplainer": {"deletion_auc": fa_gnn["deletion_auc"],
                             "insertion_auc": fa_gnn["insertion_auc"],
                             "top_nodes": gnn["top_nodes"]},
            "random_control": {"deletion_auc": fa_rnd["deletion_auc"],
                               "insertion_auc": fa_rnd["insertion_auc"]},
        })
        panels.append((denormalise(image, cfg), cam, grid, true_name,
                       classes[predicted], kind))
        cams.append(cam)
        gnn_masks.append(grid)
        print(f"  idx {index:3d} true {true_name:13s} pred {classes[predicted]:13s} "
              f"({kind})  Grad-CAM del {fa_cam['deletion_auc']:.3f} / "
              f"ins {fa_cam['insertion_auc']:.3f}  "
              f"random del {fa_rnd['deletion_auc']:.3f} / "
              f"ins {fa_rnd['insertion_auc']:.3f}", flush=True)

    for method in ("gradcam", "gnnexplainer", "random_control"):
        summary["faithfulness"][method] = {
            "deletion_auc_mean": float(np.mean([e[method]["deletion_auc"]
                                                for e in summary["examples"]])),
            "insertion_auc_mean": float(np.mean([e[method]["insertion_auc"]
                                                 for e in summary["examples"]])),
        }

    # -- figures -------------------------------------------------------------------
    fig, axes = plt.subplots(len(panels), 3, figsize=(6.6, 2.15 * len(panels)))
    axes = np.atleast_2d(axes)
    for row, (image, cam, grid, true_name, pred_name, kind) in enumerate(panels):
        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(f"{true_name}\n-> {pred_name}", fontsize=7.5,
                                color=INK if kind == "correct" else "#b91c1c")
        axes[row, 1].imshow(image)
        axes[row, 1].imshow(cam, cmap="inferno", alpha=0.45)
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(np.kron(grid, np.ones((image.shape[0] // grid.shape[0],
                                                   image.shape[1] // grid.shape[1]))),
                            cmap=SEQUENTIAL, alpha=0.5)
        for col, title in enumerate(("Preprocessed", "Grad-CAM", "GNNExplainer")):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            axes[row, col].grid(False)
            if row == 0:
                axes[row, col].set_title(title)
    fig.tight_layout()
    _save(fig, FIGURES / "xai_panel.png", REPO_ROOT / "gradcam_examples.png")

    fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels), 2.2))
    for ax, (image, cam, grid, true_name, pred_name, kind) in zip(
            np.atleast_1d(axes), panels):
        ax.imshow(image)
        ax.imshow(np.kron(grid, np.ones((image.shape[0] // grid.shape[0],
                                         image.shape[1] // grid.shape[1]))),
                  cmap=SEQUENTIAL, alpha=0.55)
        ax.set_title(f"{true_name}\n-> {pred_name}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout()
    _save(fig, FIGURES / "gnn_explainer_overlay.png",
          REPO_ROOT / "gnn_explainer_overlay.png")

    # -- LIME ----------------------------------------------------------------------
    if not args.skip_lime:
        try:
            from osteognn.xai.lime_explain import explain_image
            fig, axes = plt.subplots(2, min(3, len(panels)),
                                     figsize=(2.2 * min(3, len(panels)), 4.4))
            axes = np.atleast_2d(axes)
            lime_rows = []
            for col, (index, true_name, kind) in enumerate(examples[:3]):
                image, _ = dataset[index]
                rgb = denormalise(image.unsqueeze(0), cfg)
                result = explain_image(model, cfg, (rgb * 255).astype(np.uint8), device,
                                       num_samples=int(cfg.xai.lime_num_samples),
                                       num_features=int(cfg.xai.lime_num_features),
                                       seed=int(cfg.seed))
                axes[0, col].imshow(result["positive"])
                axes[0, col].set_title(f"{true_name}\npositive regions", fontsize=7.5)
                axes[1, col].imshow(result["signed"])
                axes[1, col].set_title("positive and negative", fontsize=7.5)
                for row in (0, 1):
                    axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
                    axes[row, col].grid(False)
                lime_rows.append({"index": index, "label": result["label"]})
            fig.tight_layout()
            _save(fig, FIGURES / "lime_bone_explanations.png",
                  REPO_ROOT / "lime_bone_explanations.png")
            summary["lime"] = lime_rows
        except Exception as exc:  # LIME is optional; its absence must not lose the rest
            print(f"  LIME skipped: {exc}", flush=True)
            summary["lime_error"] = str(exc)

    save_json(summary, RESULTS / "explainability.json")
    with open(RESULTS / "table_faithfulness.csv", "w", encoding="utf-8") as handle:
        handle.write("method,deletion_auc_mean,insertion_auc_mean\n")
        for method, values in summary["faithfulness"].items():
            handle.write(f"{method},{values['deletion_auc_mean']:.4f},"
                         f"{values['insertion_auc_mean']:.4f}\n")
    print(f"\nwrote {RESULTS/'explainability.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
