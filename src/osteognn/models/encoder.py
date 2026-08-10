"""Shared radiographic encoders.

EfficientNetB0's convolutional trunk is a stem, seven MBConv stages and a final 1x1 head
convolution -- nine top-level blocks. Rather than freezing everything or fine-tuning
wholesale, only the last six are made trainable, leaving the stem and first two MBConv
stages at their ImageNet weights: the higher-level filters adapt to radiographic texture
while the most generic low-level filters stay undisturbed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def _weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    table = {
        "efficientnet_b0": torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        "densenet121": torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
        "vgg19": torchvision.models.VGG19_Weights.IMAGENET1K_V1,
    }
    if name not in table:
        raise KeyError(f"unsupported backbone {name!r}")
    return table[name]


def build_backbone(name: str = "efficientnet_b0", pretrained: bool = True,
                   trainable_blocks: int = 6) -> tuple[nn.Module, int]:
    """Return ``(feature_extractor, out_channels)`` producing a (B, C, H, W) map."""
    weights = _weights(name, pretrained)
    if name == "efficientnet_b0":
        net = torchvision.models.efficientnet_b0(weights=weights)
        features, channels = net.features, 1280
    elif name == "resnet50":
        net = torchvision.models.resnet50(weights=weights)
        features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                 net.layer1, net.layer2, net.layer3, net.layer4)
        channels = 2048
    elif name == "densenet121":
        net = torchvision.models.densenet121(weights=weights)
        features = nn.Sequential(net.features, nn.ReLU(inplace=True))
        channels = 1024
    elif name == "vgg19":
        net = torchvision.models.vgg19(weights=weights)
        features, channels = net.features, 512
    else:
        raise KeyError(f"unsupported backbone {name!r}")
    freeze_early_blocks(features, trainable_blocks)
    return features, channels


def freeze_early_blocks(features: nn.Sequential, trainable_blocks: int) -> None:
    """Freeze all but the last ``trainable_blocks`` top-level children."""
    blocks = list(features.children())
    if trainable_blocks >= len(blocks):
        return
    cutoff = len(blocks) - trainable_blocks
    for block in blocks[:cutoff]:
        for param in block.parameters():
            param.requires_grad = False


class BackboneGAPFC(nn.Module):
    """Baseline: backbone -> global average pool -> dropout -> linear.

    This is the conventional transfer-learning classifier the literature reports, trained
    here under exactly the preprocessing, partition, augmentation and training budget of
    the proposed model so the comparison isolates architecture.
    """

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 trainable_blocks: int = 6, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features, channels = build_backbone(backbone, pretrained, trainable_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(channels, n_classes))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.features(images)
        logits = self.classifier(self.pool(feats).flatten(1))
        # Reported through the same interface as the proposed model so the evaluation
        # path is shared: no branch-specific code in evaluate().
        return {"logits": logits, "graph_logits": logits, "aux_logits": None,
                "features": feats}
