"""Deletion and insertion curves -- the quantitative check the qualitative maps lack.

A saliency map that looks anatomically plausible is not evidence that the model used
those regions. Deletion progressively removes the highest-ranked pixels and tracks the
target-class probability: a faithful explanation makes it fall fast, so a *low* deletion
AUC is good. Insertion starts from a blurred image and adds the highest-ranked pixels
back, so a *high* insertion AUC is good. Both are reported against a random-order
control, without which the absolute values are uninterpretable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _probability(model, image: torch.Tensor, class_index: int) -> float:
    with torch.no_grad():
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(image, tta_hflip=False)
        else:
            p = torch.softmax(model(image)["logits"], dim=1)
    return float(p[0, class_index])


def _blurred(image: torch.Tensor, sigma: float = 11.0) -> torch.Tensor:
    """Blurred baseline for insertion: preserves low-frequency layout, destroys texture."""
    k = int(sigma) | 1
    coords = torch.arange(k, dtype=torch.float32, device=image.device) - k // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * (sigma / 3) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(image.shape[1], 1, k, k)
    return F.conv2d(image, kernel, padding=k // 2, groups=image.shape[1])


def deletion_insertion(model, image: torch.Tensor, saliency: np.ndarray,
                       class_index: int, steps: int = 20) -> dict:
    """Both curves and their normalised areas for one image and one saliency map."""
    flat_order = np.argsort(-saliency.reshape(-1))
    n_pixels = flat_order.size
    per_step = max(1, n_pixels // steps)
    baseline = _blurred(image)

    deletion_curve, insertion_curve = [], []
    deleted = image.clone()
    inserted = baseline.clone()
    deletion_curve.append(_probability(model, deleted, class_index))
    insertion_curve.append(_probability(model, inserted, class_index))

    h, w = saliency.shape
    for step in range(steps):
        idx = flat_order[step * per_step:(step + 1) * per_step]
        rows, cols = np.unravel_index(idx, (h, w))
        deleted[0, :, rows, cols] = baseline[0, :, rows, cols]
        inserted[0, :, rows, cols] = image[0, :, rows, cols]
        deletion_curve.append(_probability(model, deleted, class_index))
        insertion_curve.append(_probability(model, inserted, class_index))

    return {
        "deletion_curve": deletion_curve,
        "insertion_curve": insertion_curve,
        "deletion_auc": float(np.trapezoid(deletion_curve, dx=1.0 / steps)),
        "insertion_auc": float(np.trapezoid(insertion_curve, dx=1.0 / steps)),
    }


def random_control(model, image: torch.Tensor, class_index: int, steps: int = 20,
                   seed: int = 0) -> dict:
    """The same curves under a random pixel order -- the null the AUCs are read against."""
    rng = np.random.RandomState(seed)
    size = image.shape[-2:]
    return deletion_insertion(model, image, rng.rand(*size), class_index, steps)
