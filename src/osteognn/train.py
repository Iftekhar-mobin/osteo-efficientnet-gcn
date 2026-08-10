"""Training loop: differential learning rates, warmup + cosine, AMP, early stopping."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .data.datasets import build_loaders, class_weights
from .models.ensemble import build_model
from .utils import get_device, set_seed


def lr_multiplier(epoch: int, warmup: int, total: int) -> float:
    """Linear warmup then a single cosine decay to zero -- no restarts (Eq. 10).

    A bare cosine at the backbone learning rate destabilises the pretrained weights in
    the first steps, which is what the warmup exists to prevent.
    """
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(model: nn.Module, cfg):
    """Two parameter groups: the unfrozen backbone updated an order of magnitude more
    slowly than the projection, graph branch and auxiliary branch."""
    if hasattr(model, "parameter_groups"):
        groups = model.parameter_groups()
    else:
        backbone, head = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (backbone if name.startswith("features.") else head).append(param)
        groups = {"backbone": backbone, "head": head}
    return torch.optim.AdamW([
        {"params": groups["backbone"], "lr": float(cfg.train.lr_backbone)},
        {"params": groups["head"], "lr": float(cfg.train.lr_head)},
    ], weight_decay=float(cfg.train.weight_decay))


def weighted_ce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                smoothing: float) -> torch.Tensor:
    return F.cross_entropy(logits, targets, weight=weights, label_smoothing=smoothing)


def compute_loss(out: dict, targets: torch.Tensor, weights: torch.Tensor, cfg) -> torch.Tensor:
    """L = CE(z_GCN, y) + lambda_aux * CE(z_CNN, y)  (Eq. 9)."""
    smoothing = float(cfg.train.label_smoothing)
    lam = float(cfg.train.aux_loss_weight)
    loss = torch.tensor(0.0, device=targets.device)
    if out["graph_logits"] is not None:
        loss = loss + weighted_ce(out["graph_logits"], targets, weights, smoothing)
    if out["aux_logits"] is not None:
        aux = weighted_ce(out["aux_logits"], targets, weights, smoothing)
        loss = loss + (lam * aux if out["graph_logits"] is not None else aux)
    return loss


@torch.no_grad()
def collect_probabilities(model: nn.Module, loader, device, tta_hflip: bool,
                          fusion_weight: float | None = None):
    """Fused probabilities and labels for a whole partition."""
    model.eval()
    probs, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(images, tta_hflip=tta_hflip, weight=fusion_weight)
        else:
            p = torch.softmax(model(images)["logits"], dim=1)
            if tta_hflip:
                flipped = torch.softmax(model(torch.flip(images, dims=[3]))["logits"], 1)
                p = 0.5 * (p + flipped)
        probs.append(p.float().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_model(cfg, out_dir: str | Path, loaders=None, verbose: bool = True) -> dict:
    """Train one configuration and return its history plus the retained checkpoint path.

    The checkpoint is selected on validation macro F1 rather than accuracy: the
    validation partition is naturally imbalanced, and accuracy alone can favour a model
    that neglects the minority Osteopenia class.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg.seed))
    device = get_device()

    loaders = loaders or build_loaders(cfg)
    classes = list(cfg.data.classes)
    model = build_model(cfg, n_classes=len(classes)).to(device)

    train_labels = loaders["train"].dataset.labels
    weights = class_weights(train_labels, len(classes)).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: lr_multiplier(e, int(cfg.train.warmup_epochs),
                                           int(cfg.train.epochs)))
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best = {"val_macro_f1": -1.0, "epoch": -1}
    patience = int(cfg.train.early_stopping_patience)
    stale = 0
    checkpoint_path = out_dir / "checkpoint.pt"
    started = time.time()

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        running, seen, correct = 0.0, 0, 0
        for images, targets in loaders["train"]:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(images)
                loss = compute_loss(out, targets, weights, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(cfg.train.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)
            correct += (out["logits"].argmax(1) == targets).sum().item()
        scheduler.step()

        # validation: deterministic pipeline, no TTA (TTA is an inference-time choice)
        model.eval()
        val_loss, val_seen, val_correct = 0.0, 0, 0
        preds, trues = [], []
        with torch.no_grad():
            for images, targets in loaders["val"]:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                out = model(images)
                val_loss += compute_loss(out, targets, weights, cfg).item() * images.size(0)
                val_seen += images.size(0)
                predicted = out["logits"].argmax(1)
                val_correct += (predicted == targets).sum().item()
                preds.append(predicted.cpu().numpy())
                trues.append(targets.cpu().numpy())
        macro_f1 = float(f1_score(np.concatenate(trues), np.concatenate(preds),
                                  average="macro", zero_division=0))

        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, seen),
            "train_acc": correct / max(1, seen),
            "val_loss": val_loss / max(1, val_seen),
            "val_acc": val_correct / max(1, val_seen),
            "val_macro_f1": macro_f1,
            "lr_backbone": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if verbose:
            print(f"epoch {record['epoch']:3d} "
                  f"train_loss {record['train_loss']:.4f} acc {record['train_acc']:.4f} | "
                  f"val_loss {record['val_loss']:.4f} acc {record['val_acc']:.4f} "
                  f"macroF1 {macro_f1:.4f}", flush=True)

        if macro_f1 > best["val_macro_f1"]:
            best = {"val_macro_f1": macro_f1, "epoch": epoch + 1,
                    "val_acc": record["val_acc"], "train_acc": record["train_acc"],
                    "val_loss": record["val_loss"], "train_loss": record["train_loss"]}
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "config": dict(cfg)}, checkpoint_path)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch + 1} "
                          f"({patience} epochs without val macro F1 improvement)",
                          flush=True)
                break

    return {
        "history": history,
        "best": best,
        "epochs_run": len(history),
        "checkpoint": str(checkpoint_path),
        "wall_time_s": time.time() - started,
        "final": history[-1] if history else None,
    }


def load_checkpoint(cfg, path: str | Path, device=None) -> nn.Module:
    device = device or get_device()
    model = build_model(cfg, n_classes=len(list(cfg.data.classes))).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model
