"""Radiographic preprocessing: CLAHE at native resolution, then resize.

The ordering is the point. Enhancing contrast *before* resampling preserves the fine
trabecular striations that carry the diagnostic signal; enhancing *after* would let
CLAHE amplify resampling artefacts instead of genuine bone structure. No denoising step
is applied, since a low-pass filter before resizing would smooth away the same texture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(class_dir: str | Path) -> list[Path]:
    """All image files in a class directory, sorted for reproducibility."""
    return sorted(
        p for p in Path(class_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def read_grayscale(path: str | Path) -> np.ndarray:
    """Read as a single-channel uint8 image, tolerating non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"could not decode image: {path}")
    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(image)


def resize(image: np.ndarray, size: int) -> np.ndarray:
    """Area interpolation when downscaling, cubic when upscaling."""
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interp)


def preprocess_image(path: str | Path, size: int = 320, clahe: bool = True,
                     clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Grayscale -> (optional) CLAHE at native resolution -> resize -> 3-channel uint8."""
    image = read_grayscale(path)
    if clahe:
        image = apply_clahe(image, clip_limit, tile_grid)
    image = resize(image, size)
    return np.repeat(image[:, :, None], 3, axis=2)


def bone_intensity_stats(paths: Iterable[str | Path]) -> dict[str, float]:
    """Mean and SD of grayscale intensity over the bone region of each radiograph.

    ``mean``/``sd`` summarise whole-image intensity. ``bone_mean``/``bone_sd`` restrict
    the statistic to pixels above an Otsu threshold, which removes the large black
    background field that would otherwise dominate and compress the between-class
    differences the analysis is meant to surface.
    """
    whole, bone = [], []
    for path in paths:
        image = read_grayscale(path)
        whole.append(float(image.mean()))
        threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = image >= threshold
        bone.append(float(image[mask].mean()) if mask.any() else float(image.mean()))
    whole_arr, bone_arr = np.asarray(whole), np.asarray(bone)
    return {
        "n": int(whole_arr.size),
        "mean": float(whole_arr.mean()),
        "sd": float(whole_arr.std(ddof=1)),
        "bone_mean": float(bone_arr.mean()),
        "bone_sd": float(bone_arr.std(ddof=1)),
        "_per_image_whole": whole_arr,
        "_per_image_bone": bone_arr,
    }
