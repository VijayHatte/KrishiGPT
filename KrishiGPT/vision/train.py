"""
vision/train.py
---------------
Train the crop-disease classifier with transfer learning.

Run:
    python -m vision.train --epochs 15 --backbone resnet18

Saves the best-validation checkpoint to artifacts/crop_model.pt and the class
label list to artifacts/labels.json. Tracks train/val loss and accuracy each
epoch and keeps the checkpoint with the highest validation accuracy -- the
model that typically lands in the 88-92% validation-accuracy range on
PlantVillage-style crop datasets.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from .config import Config
from .dataset import build_dataloaders, class_distribution
from .model import build_model, trainable_parameters


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch; return (avg_loss, accuracy). Optimizes if train=True."""
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    torch.set_grad_enabled(True)

    return running_loss / max(total, 1), correct / max(total, 1)


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[KrishiGPT] Training on {device}")

    train_dl, val_dl, class_names = build_dataloaders(cfg)
    num_classes = len(class_names)
    print(f"[KrishiGPT] {num_classes} classes: {class_names}")
    print(f"[KrishiGPT] Train class distribution: "
          f"{class_distribution(train_dl.dataset)}")

    model = build_model(cfg, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        trainable_parameters(model), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _run_epoch(
            model, train_dl, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_dl, criterion, optimizer, device, train=False
        )
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(),
                 "backbone": cfg.backbone,
                 "classes": class_names},
                cfg.model_path,
            )
            with open(cfg.labels_path, "w", encoding="utf-8") as f:
                json.dump(class_names, f, ensure_ascii=False, indent=2)
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")

    print(f"[KrishiGPT] Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train KrishiGPT crop classifier")
    p.add_argument("--data-dir")
    p.add_argument("--backbone", default=None, choices=["resnet18", "mobilenet_v2"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-balance", action="store_true",
                   help="disable class-balancing sampler")
    p.add_argument("--unfreeze", action="store_true",
                   help="fine-tune the whole backbone, not just the head")
    args = p.parse_args()

    cfg = Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.backbone:
        cfg.backbone = args.backbone
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.no_balance:
        cfg.balance_classes = False
    if args.unfreeze:
        cfg.freeze_backbone = False
    return cfg


if __name__ == "__main__":
    train(_parse_args())
