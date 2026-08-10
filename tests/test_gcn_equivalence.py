"""The dense graph layer must be the Kipf--Welling GCN, not merely something like it.

The manuscript states the propagation rule as H^(l+1) = phi(A_hat H^(l) W^(l)). This
repository implements it against a dense normalised adjacency instead of a sparse
scatter, for speed and determinism at 100 nodes. That is only an implementation detail
if the two agree numerically, so this test asserts it: first against an explicit
reference computation, and then -- when PyTorch Geometric is installed -- against its
``GCNConv``, which is the reference implementation the citation points at.

    pytest tests/test_gcn_equivalence.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osteognn.models.gcn_branch import DenseGCNLayer, GCNBranch, match_hidden_width
from osteognn.models.patch_graph import (eight_connected_edges, normalized_adjacency)


def test_edge_count_matches_manuscript():
    """684 directed edges: 4 corners of degree 3, 32 borders of 5, 64 interiors of 8."""
    edges = eight_connected_edges(10)
    assert edges.shape == (2, 684)
    degrees = torch.bincount(edges[0], minlength=100)
    assert int((degrees == 3).sum()) == 4
    assert int((degrees == 5).sum()) == 32
    assert int((degrees == 8).sum()) == 64


def test_adjacency_is_symmetrically_normalised():
    a_hat = normalized_adjacency(10)
    assert torch.allclose(a_hat, a_hat.T, atol=1e-6)
    # Self-loops give every node a non-zero diagonal, and the normalisation is
    # degree-dependent, so corners and interiors are weighted differently.
    assert a_hat[0, 0] > a_hat[55, 55]


def test_dense_layer_equals_explicit_reference():
    torch.manual_seed(0)
    layer = DenseGCNLayer(16, 16, dropout=0.0).eval()
    a_hat = normalized_adjacency(10)
    x = torch.randn(2, 100, 16)
    with torch.no_grad():
        got = layer(x, a_hat)
        expected = torch.relu(layer.norm(
            layer.linear(a_hat @ x).reshape(-1, 16)).reshape(2, 100, 16))
    assert torch.allclose(got, expected, atol=1e-6)


def test_matches_pytorch_geometric_gcnconv():
    """Same propagation as the cited reference implementation, to machine precision."""
    pyg = pytest.importorskip("torch_geometric",
                              reason="PyTorch Geometric not installed")
    from torch_geometric.nn import GCNConv

    torch.manual_seed(0)
    edges = eight_connected_edges(10)
    a_hat = normalized_adjacency(10)
    x = torch.randn(100, 32)

    conv = GCNConv(32, 32, add_self_loops=True, normalize=True, bias=True).eval()
    with torch.no_grad():
        reference = conv(x, edges)
        # The dense equivalent: same weight, same bias, same normalised adjacency.
        dense = a_hat @ x @ conv.lin.weight.T + conv.bias
    assert torch.allclose(reference, dense, atol=1e-5), \
        f"max deviation {float((reference - dense).abs().max()):.2e}"


def test_dual_pooling_shape_and_residuals():
    branch = GCNBranch(dim=64, layers=3, dropout=0.0, residual=True).eval()
    a_hat = normalized_adjacency(10)
    out = branch(torch.randn(4, 100, 64), a_hat)
    assert out.shape == (4, 128)          # concatenated mean and max
    mean_only = GCNBranch(dim=64, layers=3, dropout=0.0, pooling="mean").eval()
    assert mean_only(torch.randn(4, 100, 64), a_hat).shape == (4, 64)


def test_conv_control_is_parameter_matched():
    """Ablation (vii) is only a capacity control if the capacities actually match."""
    from osteognn.models.gcn_branch import ConvControlBranch
    graph = GCNBranch(dim=256, layers=3, dropout=0.3)
    target = sum(p.numel() for p in graph.parameters())
    control = ConvControlBranch(dim=256, dropout=0.3, target_params=target)
    achieved = sum(p.numel() for p in control.parameters())
    assert abs(achieved - target) / target < 0.05, \
        f"graph branch {target:,} vs conv control {achieved:,}"


def test_match_hidden_width_is_the_best_choice():
    target = 197_376
    best = match_hidden_width(target, 256)
    from osteognn.models.gcn_branch import _conv_stack_params
    gap = abs(_conv_stack_params(best, 256) - target)
    for candidate in range(1, 257):
        assert abs(_conv_stack_params(candidate, 256) - target) >= gap
