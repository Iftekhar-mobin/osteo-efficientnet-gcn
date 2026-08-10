"""Offline class-balancing augmentation (training partition only) and online transforms.

Every original training image is retained unchanged; each class is then topped up with
augmented variants, cycling back through the originals as needed, until it reaches
``target_per_class``. Validation and test are never augmented -- they pass through the
deterministic resize + normalise pipeline only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # albumentations is optional at import time so the module can be inspected offline
    import albumentations as A
    _HAVE_ALBUMENTATIONS = True
except Exception:  # pragma: no cover
    A = None
    _HAVE_ALBUMENTATIONS = False

import torch
from torchvision import transforms


def build_offline_pipeline(cfg) -> "A.Compose":
    """Albumentations pipeline used to manufacture the balancing variants.

    Each transform fires independently at its own probability rather than exactly one
    per image, so an augmented sample may combine several perturbations.
    """
    if not _HAVE_ALBUMENTATIONS:
        raise ImportError("albumentations is required for offline augmentation")
    scale = tuple(cfg.affine_scale)
    translate = float(cfg.affine_translate)
    rotate = float(cfg.affine_rotate_deg)
    return A.Compose([
        A.HorizontalFlip(p=float(cfg.hflip_p)),
        A.Affine(
            scale=(scale[0], scale[1]),
            translate_percent={"x": (-translate, translate), "y": (-translate, translate)},
            rotate=(-rotate, rotate),
            p=float(cfg.affine_p),
        ),
        A.RandomBrightnessContrast(p=float(cfg.brightness_contrast_p)),
        A.GaussNoise(p=float(cfg.gauss_noise_p)),
    ])


def balance_partition(items: list[tuple[str, str]], classes: list[str],
                      target_per_class: int, seed: int) -> list[tuple[str, str, int]]:
    """Return ``(source_path, label, aug_index)``; ``aug_index == 0`` means untouched.

    The plan is produced without touching pixels, so it can be inspected, committed and
    diffed. ``scripts/01`` materialises it.
    """
    rng = np.random.RandomState(seed)
    plan: list[tuple[str, str, int]] = []
    for class_name in classes:
        originals = [p for p, lab in items if lab == class_name]
        plan.extend((p, class_name, 0) for p in originals)
        deficit = target_per_class - len(originals)
        if deficit <= 0:
            continue
        if not originals:
            raise ValueError(f"class {class_name!r} has no training images to augment")
        order = rng.permutation(len(originals))
        for k in range(deficit):
            source = originals[order[k % len(originals)]]
            plan.append((source, class_name, k // len(originals) + 1))
    return plan


def build_online_transform(cfg, train: bool) -> transforms.Compose:
    """Torchvision transforms applied at every training step.

    Hue and saturation jitter are deliberately excluded: the radiographs are grayscale
    replicated across three channels, and colour perturbation would desynchronise the
    channels into hues that cannot occur in a real radiograph.
    """
    size = int(cfg.preprocess.image_size)
    mean, std = list(cfg.preprocess.norm_mean), list(cfg.preprocess.norm_std)
    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True),
            transforms.Normalize(mean, std),
        ])
    online = cfg.augment.online
    steps = [
        transforms.ToTensor(),
        transforms.RandomResizedCrop(
            size, scale=tuple(online["random_resized_crop_scale"]), antialias=True),
    ]
    if online.get("hflip", True):
        steps.append(transforms.RandomHorizontalFlip())
    steps += [
        transforms.RandomRotation(float(online["rotate_deg"])),
        transforms.ColorJitter(brightness=float(online["color_jitter_brightness"]),
                               contrast=float(online["color_jitter_contrast"])),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=float(online["random_erasing_p"])),
    ]
    return transforms.Compose(steps)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a deterministic, distinct stream."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)
