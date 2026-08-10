"""Determinism, device selection, and small IO helpers shared across the pipeline."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches.

    ``deterministic`` also pins cuDNN's algorithm choice. Convolution autotuning is the
    single largest source of run-to-run drift on identical inputs, so it is disabled by
    default: reproducing a reported number matters more here than the few percent of
    throughput autotuning would buy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_report() -> dict[str, Any]:
    """Everything needed to attribute a number to the machine that produced it."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["cuda"] = torch.version.cuda
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import albumentations
        info["albumentations"] = albumentations.__version__
    except Exception:
        pass
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except Exception:
        pass
    try:
        import psutil
        info["system_ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    return info


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return str(value)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=default)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def parameter_breakdown(model: torch.nn.Module, groups: dict[str, str]) -> dict[str, int]:
    """Sum parameters whose qualified name starts with each prefix in ``groups``."""
    out = {name: 0 for name in groups}
    for qualified_name, param in model.named_parameters():
        for name, prefix in groups.items():
            if qualified_name.startswith(prefix):
                out[name] += param.numel()
                break
    return out
