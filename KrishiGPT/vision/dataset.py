"""
vision/dataset.py
-----------------
Data loading, preprocessing and class balancing for the crop classifier.

Responsibilities (maps to the resume bullet "Preprocessed and balanced crop
image datasets"):

  * Preprocessing  -- resize / crop to 224px, tensor conversion, ImageNet
                      normalisation.
  * Augmentation   -- random flips, rotations and colour jitter on the train
                      split to improve generalisation.
  * Balancing      -- a WeightedRandomSampler that oversamples minority
                      classes so each batch is roughly class-balanced, which
                      directly reduces bias toward majority classes.
"""

from __future__ import annotations

import collections

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from .config import Config


def build_transforms(cfg: Config):
    """Return (train_tf, eval_tf) preprocessing pipelines."""
    train_tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(cfg.norm_mean, cfg.norm_std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.norm_mean, cfg.norm_std),
        ]
    )
    return train_tf, eval_tf


def _make_balanced_sampler(dataset) -> WeightedRandomSampler:
    """Build a sampler that oversamples minority classes.

    Weight of each sample = 1 / (frequency of its class). This equalises the
    expected number of samples drawn per class within an epoch.
    """
    counts = collections.Counter(label for _, label in dataset.samples)
    class_weights = {cls: 1.0 / n for cls, n in counts.items()}
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(cfg: Config):
    """Create train/val dataloaders and return (train_dl, val_dl, class_names).

    The train loader uses a class-balancing sampler when ``cfg.balance_classes``
    is True; the val loader is always sequential/unshuffled for stable metrics.
    """
    import os

    train_tf, eval_tf = build_transforms(cfg)

    train_ds = datasets.ImageFolder(
        os.path.join(cfg.data_dir, cfg.train_split), transform=train_tf
    )
    val_ds = datasets.ImageFolder(
        os.path.join(cfg.data_dir, cfg.val_split), transform=eval_tf
    )

    if cfg.balance_classes:
        sampler = _make_balanced_sampler(train_ds)
        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_dl, val_dl, train_ds.classes


def class_distribution(dataset) -> dict:
    """Return {class_name: count} -- handy for logging dataset balance."""
    counts = collections.Counter(label for _, label in dataset.samples)
    return {dataset.classes[idx]: n for idx, n in sorted(counts.items())}
