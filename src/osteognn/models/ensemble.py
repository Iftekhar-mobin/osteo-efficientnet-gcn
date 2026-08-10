"""EfficientNet-GCN: the assembled framework.

    F      = E_theta(X)                       shared 1280x10x10 feature map
    F'     = Proj(F)                          256x10x10
    G      = PatchGraph(F')                   100 nodes, 8-connectivity
    z_GCN  = DualPool(GCN_3(G))               512-d
    z_CNN  = MLP_aux(GAP(F))                  graph-free path
    P      = w*softmax(head(z_GCN)) + (1-w)*softmax(z_CNN),  w = 0.6

The two branches deliberately consume different tensors: the graph branch reasons over
the channel-reduced F', while the auxiliary branch pools the full, un-reduced F.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import build_backbone
from .gcn_branch import ConvControlBranch, GCNBranch
from .patch_graph import PatchGraphConstruction


def _mlp(dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.ReLU(inplace=True), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class EfficientNetGCN(nn.Module):
    def __init__(self, cfg, n_classes: int = 3):
        super().__init__()
        model_cfg = cfg.model
        self.use_gcn = bool(model_cfg.use_gcn_branch)
        self.use_aux = bool(model_cfg.use_aux_branch)
        self.conv_control = bool(model_cfg.get("conv_control", False))
        if not self.use_gcn and not self.use_aux:
            raise ValueError("at least one branch must be enabled")

        self.features, channels = build_backbone(
            model_cfg.backbone, bool(model_cfg.pretrained),
            int(model_cfg.trainable_blocks))
        self.backbone_channels = channels
        graph_cfg, gcn_cfg = model_cfg.graph, model_cfg.gcn
        dropout = float(model_cfg.head["dropout"])

        if self.use_gcn:
            self.patch_graph = PatchGraphConstruction(
                in_channels=channels, out_channels=int(model_cfg.proj_channels),
                grid=int(graph_cfg["grid"]), self_loops=bool(graph_cfg["self_loops"]),
                symmetric=bool(graph_cfg["symmetric_norm"]))
            branch_kwargs = dict(dim=int(model_cfg.proj_channels),
                                 dropout=float(gcn_cfg["dropout"]),
                                 pooling=str(gcn_cfg["pooling"]))
            if self.conv_control:
                reference = GCNBranch(layers=int(gcn_cfg["layers"]), residual=True,
                                      **branch_kwargs)
                target = sum(p.numel() for p in reference.parameters())
                self.graph_branch = ConvControlBranch(target_params=target,
                                                      **branch_kwargs)
            else:
                self.graph_branch = GCNBranch(
                    layers=int(gcn_cfg["layers"]),
                    residual=bool(gcn_cfg["residual"]), **branch_kwargs)
            head_dims = list(model_cfg.head["graph_mlp"])
            head_dims[0] = self.graph_branch.out_dim
            head_dims[-1] = n_classes
            self.graph_head = _mlp(head_dims, dropout)

        if self.use_aux:
            aux_dims = list(model_cfg.head["aux_mlp"])
            aux_dims[0], aux_dims[-1] = channels, n_classes
            self.aux_pool = nn.AdaptiveAvgPool2d(1)
            self.aux_head = _mlp(aux_dims, dropout)

        self.fusion_weight = float(cfg.inference.fusion_weight)

    # -- forward ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.features(images)
        graph_logits = aux_logits = None

        if self.use_gcn:
            nodes, projected = self.patch_graph(feats)
            if self.conv_control:
                embedding = self.graph_branch(projected)
            else:
                # The graph propagation runs in full precision: its repeated
                # normalised aggregations are where fp16 accumulation error shows up.
                with torch.autocast(device_type=images.device.type, enabled=False):
                    embedding = self.graph_branch(nodes.float(),
                                                  self.patch_graph.a_hat.float())
            graph_logits = self.graph_head(embedding)

        if self.use_aux:
            aux_logits = self.aux_head(self.aux_pool(feats).flatten(1))

        return {"graph_logits": graph_logits, "aux_logits": aux_logits,
                "features": feats,
                "logits": graph_logits if graph_logits is not None else aux_logits}

    # -- inference ----------------------------------------------------------------
    def fuse(self, out: dict[str, torch.Tensor], weight: float | None = None) -> torch.Tensor:
        """Weighted probability fusion of the two branches (Eq. 8)."""
        w = self.fusion_weight if weight is None else weight
        graph, aux = out["graph_logits"], out["aux_logits"]
        if graph is None:
            return torch.softmax(aux, dim=1)
        if aux is None or w >= 1.0:
            return torch.softmax(graph, dim=1)
        return w * torch.softmax(graph, dim=1) + (1.0 - w) * torch.softmax(aux, dim=1)

    def predict_proba(self, images: torch.Tensor, tta_hflip: bool = True,
                      weight: float | None = None) -> torch.Tensor:
        """Fused class probabilities, optionally averaged with the horizontal mirror."""
        probs = self.fuse(self(images), weight)
        if tta_hflip:
            probs = 0.5 * (probs + self.fuse(self(torch.flip(images, dims=[3])), weight))
        return probs

    # -- introspection ------------------------------------------------------------
    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Backbone parameters versus everything else, for differential learning rates."""
        backbone, head = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (backbone if name.startswith("features.") else head).append(param)
        return {"backbone": backbone, "head": head}

    def component_parameters(self) -> dict[str, int]:
        prefixes = {
            "backbone": "features.",
            "projection": "patch_graph.proj",
            "graph_branch": "graph_branch.",
            "graph_head": "graph_head.",
            "aux_branch": "aux_head.",
        }
        out = {k: 0 for k in prefixes}
        for name, param in self.named_parameters():
            for key, prefix in prefixes.items():
                if name.startswith(prefix):
                    out[key] += param.numel()
                    break
        return out


class SarhanProtocolModel(EfficientNetGCN):
    """Identical architecture; exists only so the protocol probe is named explicitly.

    Sarhan et al. partition *after* augmentation. The probe in
    ``scripts/05_baselines.py`` trains this same network under that ordering to measure
    how much of the reported margin the ordering alone can produce. It is a test of the
    protocol, not a reimplementation of their network, which is not specified in enough
    detail to reproduce faithfully.
    """


def build_model(cfg, n_classes: int = 3) -> nn.Module:
    name = str(cfg.model.get("name", "efficientnet_gcn"))
    if name == "efficientnet_gcn":
        return EfficientNetGCN(cfg, n_classes)
    if name == "backbone_gap_fc":
        from .encoder import BackboneGAPFC
        return BackboneGAPFC(str(cfg.model.backbone), bool(cfg.model.pretrained),
                             int(cfg.model.trainable_blocks), n_classes,
                             float(cfg.model.head["dropout"]))
    raise KeyError(f"unknown model {name!r}")
