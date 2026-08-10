"""Grad-CAM over the final EfficientNetB0 block, driven by the graph branch's logits.

This differs from the standard application of Grad-CAM to a convolutional classifier:
gradients propagate from z_GCN back through the classification head, the three graph
layers and the 1x1 projection before reaching the target activation map, so the saliency
reflects the graph branch's use of the backbone features rather than a purely
convolutional decision path.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer: torch.nn.Module, branch: str = "graph"):
        self.model = model.eval()
        self.branch = branch
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        for handle in self._handles:
            handle.remove()

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> tuple:
        """Return ``(cam, class_index)`` with ``cam`` normalised to [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        out = self.model(image)
        key = "graph_logits" if (self.branch == "graph" and
                                 out.get("graph_logits") is not None) else "logits"
        logits = out[key]
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[0, class_index].backward(retain_graph=False)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # GAP over gradients
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, class_index


def resolve_layer(model, dotted: str) -> torch.nn.Module:
    """Resolve e.g. ``features.8`` to the module object."""
    node = model
    for part in dotted.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node
