"""Torch datasets over the preprocessed partitions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .augment import build_online_transform, seed_worker


class RadiographDataset(Dataset):
    """Reads preprocessed 320x320 PNGs written by ``scripts/01_preprocess_and_split.py``.

    Preprocessing is materialised to disk rather than done on the fly because CLAHE at
    native resolution is the expensive step and it is deterministic: doing it once keeps
    every epoch, every ablation and every baseline reading byte-identical inputs.
    """

    def __init__(self, records: list[dict], classes: list[str], cfg, train: bool):
        self.records = records
        self.classes = list(classes)
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}
        self.transform = build_online_transform(cfg, train=train)
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        tensor = self.transform(image)
        label = self.class_to_index[record["label"]]
        return tensor, label

    @property
    def labels(self) -> np.ndarray:
        return np.array([self.class_to_index[r["label"]] for r in self.records])


def load_partition(processed_dir: str | Path, partition: str) -> list[dict]:
    """Read the per-partition index written during preprocessing."""
    import json
    path = Path(processed_dir) / f"{partition}_index.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_loaders(cfg, partitions=("train", "val", "test")) -> dict[str, DataLoader]:
    classes = list(cfg.data.classes)
    loaders: dict[str, DataLoader] = {}
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    for partition in partitions:
        records = load_partition(cfg.data.processed_dir, partition)
        train = partition == "train"
        dataset = RadiographDataset(records, classes, cfg, train=train)
        loaders[partition] = DataLoader(
            dataset,
            batch_size=int(cfg.train.batch_size),
            shuffle=train,
            num_workers=int(cfg.train.num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator if train else None,
            persistent_workers=int(cfg.train.num_workers) > 0,
        )
    return loaders


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    On the balanced training partition these are uniform by construction; they are
    retained so the identical objective applies unchanged to ablation (v), which trains
    on the raw 546/261/555 distribution.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)
