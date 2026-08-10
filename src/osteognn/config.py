"""Typed-ish configuration tree, loadable from YAML and overridable from the CLI."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """A dict that also supports attribute access and dotted get/set."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def copy_with(self, **overrides: Any) -> "Config":
        out = Config(copy.deepcopy(dict(self)))
        for dotted, value in overrides.items():
            out.set_path(dotted.replace("__", "."), value)
        return out

    def to_json(self) -> str:
        return json.dumps(dict(self), indent=2, sort_keys=True, default=str)


def _coerce(text: str) -> Any:
    """Parse a CLI override value with YAML semantics (true/1.0e-4/[1,2]/str)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: list[str] | None = None,
                **kwargs: Any) -> Config:
    """Load YAML, then apply ``key.path=value`` strings, then keyword overrides."""
    with open(path, "r", encoding="utf-8") as handle:
        cfg = Config(yaml.safe_load(handle))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        dotted, _, raw = item.partition("=")
        cfg.set_path(dotted.strip(), _coerce(raw.strip()))
    for dotted, value in kwargs.items():
        cfg.set_path(dotted.replace("__", "."), value)
    return cfg


# --------------------------------------------------------------------------------------
# Ablation and baseline variants.
#
# Each entry is a set of dotted overrides applied to the default config. Everything not
# listed is held fixed, which is what makes the ablation table a controlled comparison:
# identical splits, identical backbone, identical training budget.
# --------------------------------------------------------------------------------------
ABLATIONS: dict[str, dict[str, Any]] = {
    # (i) patch graph and its three GCN layers removed; auxiliary branch alone predicts
    "no_gcn": {"model.use_gcn_branch": False, "run_name": "abl_i_no_gcn"},
    # (ii) fusion replaced by the graph branch's softmax alone (inference-time only)
    "no_aux": {"inference.fusion_weight": 1.0, "run_name": "abl_ii_no_aux",
               "_inference_only": True},
    # (iii) concatenated mean--max readout replaced by mean pooling only
    "mean_pool": {"model.gcn.pooling": "mean", "model.head.graph_mlp": [256, 256, 128, 3],
                  "run_name": "abl_iii_mean_pool"},
    # (iv) CLAHE contrast enhancement skipped
    "no_clahe": {"preprocess.clahe": False, "run_name": "abl_iv_no_clahe"},
    # (v) trained on the original 546/261/555 distribution
    "no_balance": {"augment.balance": False, "run_name": "abl_v_no_balance"},
    # (vi) inference uses a single forward pass (inference-time only)
    "no_tta": {"inference.tta_hflip": False, "run_name": "abl_vi_no_tta",
               "_inference_only": True},
    # (vii) graph branch replaced by a parameter-matched 3x3 convolutional block
    "conv_control": {"model.conv_control": True, "run_name": "abl_vii_conv_control"},
}

BASELINES: dict[str, dict[str, Any]] = {
    "effb0_gap_fc": {"model.name": "backbone_gap_fc", "model.backbone": "efficientnet_b0",
                     "run_name": "base_effb0"},
    "resnet50": {"model.name": "backbone_gap_fc", "model.backbone": "resnet50",
                 "run_name": "base_resnet50"},
    "densenet121": {"model.name": "backbone_gap_fc", "model.backbone": "densenet121",
                    "run_name": "base_densenet121"},
    "vgg19": {"model.name": "backbone_gap_fc", "model.backbone": "vgg19",
              "run_name": "base_vgg19"},
}

# The protocol probe is not a baseline architecture. Sarhan et al. (2024) report 97.50%
# on this same corpus but partition *after* augmentation; their network is not specified
# in enough detail to reimplement faithfully, so guessing at it would produce a number
# that measures our guess rather than their method. What can be isolated cleanly is the
# ordering itself: this variant trains the *proposed* model, unchanged, under the
# augment-then-split ordering, so the difference between it and the full model is
# attributable to the protocol and nothing else.
PROTOCOL_PROBE = {
    "leaky_split": {"_augment_before_split": True, "run_name": "probe_leaky_split"},
}


def variant_config(base: Config, name: str) -> Config:
    """Return ``base`` with the named ablation, baseline or probe overrides applied."""
    table = next((t for t in (ABLATIONS, BASELINES, PROTOCOL_PROBE) if name in t), {})
    if name not in table:
        raise KeyError(
            f"unknown variant {name!r}; known: "
            f"{sorted(ABLATIONS) + sorted(BASELINES) + sorted(PROTOCOL_PROBE)}")
    overrides = {k: v for k, v in table[name].items() if not k.startswith("_")}
    return base.copy_with(**{k.replace(".", "__"): v for k, v in overrides.items()})


def is_inference_only(name: str) -> bool:
    """True when the variant is a re-evaluation of the full model, not a retrain."""
    return bool(ABLATIONS.get(name, {}).get("_inference_only", False))
