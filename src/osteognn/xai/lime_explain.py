"""LIME super-pixel explanations over the fused output probabilities."""
from __future__ import annotations

import numpy as np
import torch


def make_predict_fn(model, cfg, device):
    """Batch predictor LIME can call with (H, W, 3) uint8/float arrays."""
    mean = np.array(cfg.preprocess.norm_mean, dtype=np.float32)
    std = np.array(cfg.preprocess.norm_std, dtype=np.float32)

    def predict(images: np.ndarray) -> np.ndarray:
        batch = images.astype(np.float32)
        if batch.max() > 1.5:
            batch = batch / 255.0
        batch = (batch - mean) / std
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
        out = []
        with torch.no_grad():
            for start in range(0, tensor.shape[0], 32):
                chunk = tensor[start:start + 32]
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(chunk, tta_hflip=False)
                else:
                    p = torch.softmax(model(chunk)["logits"], dim=1)
                out.append(p.float().cpu().numpy())
        return np.concatenate(out)

    return predict


def explain_image(model, cfg, image_rgb: np.ndarray, device, num_samples: int = 1000,
                  num_features: int = 8, seed: int = 0) -> dict:
    """Return positive-only and signed super-pixel explanations for the top class."""
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    explainer = lime_image.LimeImageExplainer(random_state=seed)
    predict = make_predict_fn(model, cfg, device)
    explanation = explainer.explain_instance(
        image_rgb.astype(np.double), predict, top_labels=3, hide_color=0,
        num_samples=num_samples, random_seed=seed)
    label = explanation.top_labels[0]
    pos_img, pos_mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=num_features, hide_rest=False)
    both_img, both_mask = explanation.get_image_and_mask(
        label, positive_only=False, num_features=num_features, hide_rest=False)
    return {
        "label": int(label),
        "positive": mark_boundaries(pos_img / 255.0, pos_mask),
        "signed": mark_boundaries(both_img / 255.0, both_mask),
        "positive_mask": pos_mask,
        "signed_mask": both_mask,
    }
