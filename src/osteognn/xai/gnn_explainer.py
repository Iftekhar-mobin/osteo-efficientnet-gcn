"""GNNExplainer over the patch graph: a learned soft mask on node features.

Only the node-feature mask is reported. GNNExplainer also produces an edge mask, but the
adjacency here is fixed and identical for every input image, so an edge mask is defined
over a topology that does not vary across images and carries no per-image information.

The objective follows Ying et al.: maximise the target-class log-probability under the
masked node features, regularised by mask size and element-wise entropy so the
explanation is driven to be both small and near-binary.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def explain_nodes(model, image: torch.Tensor, class_index: int | None = None,
                  epochs: int = 200, lr: float = 0.01, size_coeff: float = 0.005,
                  entropy_coeff: float = 1.0, seed: int = 0) -> dict:
    """Learn a per-node importance mask for one image.

    Returns the mask over the 100 patches, reshaped to the 10x10 grid.
    """
    model.eval()
    device = image.device
    torch.manual_seed(seed)

    with torch.no_grad():
        feats = model.features(image)
        nodes, _ = model.patch_graph(feats)          # (1, N, C)
        base_logits = model.graph_head(
            model.graph_branch(nodes.float(), model.patch_graph.a_hat.float()))
        if class_index is None:
            class_index = int(base_logits.argmax(dim=1).item())

    n_nodes = nodes.shape[1]
    # Small random init breaks the symmetry between structurally identical patches.
    mask_logits = torch.nn.Parameter(
        torch.randn(n_nodes, device=device) * 0.1)
    optimizer = torch.optim.Adam([mask_logits], lr=lr)

    history = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        mask = torch.sigmoid(mask_logits).view(1, n_nodes, 1)
        masked = nodes.float() * mask
        logits = model.graph_head(
            model.graph_branch(masked, model.patch_graph.a_hat.float()))
        log_prob = F.log_softmax(logits, dim=1)[0, class_index]
        m = torch.sigmoid(mask_logits)
        size_penalty = size_coeff * m.sum()
        entropy = -(m * torch.log(m + 1e-9) + (1 - m) * torch.log(1 - m + 1e-9)).mean()
        loss = -log_prob + size_penalty + entropy_coeff * entropy
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    mask = torch.sigmoid(mask_logits).detach().cpu().numpy()
    grid = int(np.sqrt(n_nodes))
    normalised = ((mask - mask.min()) / (mask.max() - mask.min())
                  if mask.max() > mask.min() else np.zeros_like(mask))
    return {
        "node_mask": mask,
        "node_mask_grid": mask.reshape(grid, grid),
        "node_mask_normalised": normalised.reshape(grid, grid),
        "class_index": class_index,
        "loss_history": history,
        "top_nodes": np.argsort(-mask)[:10].tolist(),
    }
