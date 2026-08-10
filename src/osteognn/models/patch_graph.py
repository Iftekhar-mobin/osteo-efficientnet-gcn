"""Patch-Graph Construction Module.

A 1x1 convolution channel-reduces the shared feature map from 1,280 to 256 channels,
then each of the 100 spatial positions of the reduced map becomes one node of an
eight-connected patch graph. The projection exists to control the graph branch's
parameter count: a graph layer consuming the raw 1,280-channel features would need a
1280x1280 weight matrix (~1.64M parameters in a single layer), whereas the 1x1
convolution performs the reduction with 0.33M and leaves each graph layer at 256x256
(65.5K).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def eight_connected_edges(grid: int) -> torch.Tensor:
    """Directed edge index (2, E) for an eight-connected ``grid`` x ``grid`` lattice.

    For grid=10 this yields 684 directed edges: 4 corner patches of degree 3, 32 border
    patches of degree 5 and 64 interior patches of degree 8.
    """
    src, dst = [], []
    for row in range(grid):
        for col in range(grid):
            node = row * grid + col
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    r, c = row + d_row, col + d_col
                    if 0 <= r < grid and 0 <= c < grid:
                        src.append(node)
                        dst.append(r * grid + c)
    return torch.tensor([src, dst], dtype=torch.long)


def normalized_adjacency(grid: int, self_loops: bool = True,
                         symmetric: bool = True) -> torch.Tensor:
    """Return the dense propagation matrix A_hat = D~^-1/2 (A + I) D~^-1/2.

    Because the degree varies with position, A_hat weights border and interior patches
    differently -- the property that distinguishes this propagation from a 3x3
    convolution, which applies an identical kernel everywhere and handles boundaries by
    padding.
    """
    n = grid * grid
    edges = eight_connected_edges(grid)
    adjacency = torch.zeros(n, n, dtype=torch.float32)
    adjacency[edges[0], edges[1]] = 1.0
    if self_loops:
        adjacency = adjacency + torch.eye(n, dtype=torch.float32)
    if not symmetric:
        return adjacency
    degree = adjacency.sum(dim=1)
    d_inv_sqrt = degree.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.unsqueeze(1) * adjacency * d_inv_sqrt.unsqueeze(0)


class PatchGraphConstruction(nn.Module):
    """F -> F' = phi(BN(Conv1x1(F))), reshaped to a (B, N, C) node set."""

    def __init__(self, in_channels: int = 1280, out_channels: int = 256, grid: int = 10,
                 self_loops: bool = True, symmetric: bool = True):
        super().__init__()
        self.grid = grid
        self.n_nodes = grid * grid
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.register_buffer("a_hat", normalized_adjacency(grid, self_loops, symmetric))
        self.register_buffer("edge_index", eight_connected_edges(grid))

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(nodes, projected_map)`` with nodes shaped (B, N, C)."""
        projected = self.proj(features)                      # (B, C, G, G)
        b, c, h, w = projected.shape
        if h * w != self.n_nodes:
            raise ValueError(
                f"feature map is {h}x{w} = {h * w} positions but the graph expects "
                f"{self.n_nodes}; check preprocess.image_size")
        nodes = projected.flatten(2).transpose(1, 2).contiguous()   # (B, N, C)
        return nodes, projected
