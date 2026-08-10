"""Residual Dual-Pooling Graph Reasoning Branch, and its parameter-matched conv control.

The graph layer is the first-order localised spectral propagation rule of Kipf and
Welling, H^(l+1) = phi(A_hat H^(l) W^(l)), implemented against a dense normalised
adjacency rather than a sparse scatter. At 100 nodes the dense form is a single batched
matmul -- faster, deterministic, and dependency-free. ``tests/test_gcn_equivalence.py``
checks it against PyTorch Geometric's ``GCNConv`` to machine precision, so the choice is
an implementation detail rather than a change of model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DenseGCNLayer(nn.Module):
    """One propagation step: X -> A_hat X W, then batch norm, ReLU, dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3,
                 bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # nodes: (B, N, C_in); a_hat: (N, N)
        propagated = torch.matmul(a_hat, nodes)           # neighbourhood aggregation
        transformed = self.linear(propagated)
        b, n, c = transformed.shape
        normed = self.norm(transformed.reshape(b * n, c)).reshape(b, n, c)
        return self.dropout(self.act(normed))


class GCNBranch(nn.Module):
    """Three stacked 256->256 graph layers with residuals around layers 2 and 3.

    Stacking three eight-connected propagations widens the effective receptive field
    from 3x3 to 7x7 patches. The readout concatenates mean pooling (the joint-wide
    density summary) with max pooling (the response of the single most abnormal patch).
    """

    def __init__(self, dim: int = 256, layers: int = 3, dropout: float = 0.3,
                 residual: bool = True, pooling: str = "mean_max"):
        super().__init__()
        if pooling not in {"mean_max", "mean"}:
            raise ValueError(f"unsupported pooling {pooling!r}")
        self.layers = nn.ModuleList(
            DenseGCNLayer(dim, dim, dropout) for _ in range(layers))
        self.residual = residual
        self.pooling = pooling
        self.out_dim = dim * (2 if pooling == "mean_max" else 1)

    def forward(self, nodes: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        hidden = nodes
        for index, layer in enumerate(self.layers):
            out = layer(hidden, a_hat)
            # residual connections around the second and third layers only
            hidden = out + hidden if (self.residual and index >= 1) else out
        self.node_features = hidden  # retained for GNNExplainer
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        return torch.cat([hidden.mean(dim=1), hidden.max(dim=1).values], dim=1)


def _conv_stack_params(hidden: int, dim: int) -> int:
    """Parameter count of the 3-layer 3x3 conv stack used by the capacity control."""
    return (9 * dim * hidden + hidden) + (9 * hidden * hidden + hidden) + \
           (9 * hidden * dim + dim)


def match_hidden_width(target_params: int, dim: int = 256) -> int:
    """Smallest 3x3-conv hidden width whose parameter count is closest to ``target``."""
    best, best_gap = 1, float("inf")
    for hidden in range(1, dim + 1):
        gap = abs(_conv_stack_params(hidden, dim) - target_params)
        if gap < best_gap:
            best, best_gap = hidden, gap
    return best


class ConvControlBranch(nn.Module):
    """Ablation (vii): the graph branch replaced by a parameter-matched conv block.

    Because the patch adjacency is fixed, the graph propagation aggregates over a 3x3
    spatial neighbourhood at each layer, so any accuracy gain attributed to relational
    reasoning must be shown to exceed what an equally sized convolutional block achieves
    on the same projected features. Without this control, removing the graph branch
    alone cannot distinguish a benefit of graph structure from a benefit of added
    parameters. The hidden width is chosen automatically so the two branches match to
    within a fraction of a percent; the achieved counts are reported in the results.
    """

    def __init__(self, dim: int = 256, dropout: float = 0.3, pooling: str = "mean_max",
                 target_params: int | None = None):
        super().__init__()
        if target_params is None:
            reference = GCNBranch(dim=dim, dropout=dropout, pooling=pooling)
            target_params = sum(p.numel() for p in reference.parameters())
        hidden = match_hidden_width(target_params, dim)
        self.hidden_width = hidden
        self.matched_target = target_params
        self.block = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(hidden, dim, 3, padding=1), nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        self.pooling = pooling
        self.out_dim = dim * (2 if pooling == "mean_max" else 1)

    def forward(self, projected: torch.Tensor, a_hat: torch.Tensor | None = None):
        out = self.block(projected)                       # (B, C, G, G)
        flat = out.flatten(2)                             # (B, C, N)
        if self.pooling == "mean":
            return flat.mean(dim=2)
        return torch.cat([flat.mean(dim=2), flat.max(dim=2).values], dim=1)
