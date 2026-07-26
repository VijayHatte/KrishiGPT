"""
vision/evaluate.py
------------------
Evaluate the trained crop classifier (resume bullet: "Evaluated performance
using precision, recall, F1-score, and confusion matrix; conducted error
analysis to reduce false positives").

Run:
    python -m vision.evaluate

Produces:
  * a per-class classification report (precision / recall / F1 / support),
  * a confusion matrix saved as artifacts/confusion_matrix.png,
  * an error-analysis table of the most-confused class pairs, written to
    artifacts/error_analysis.csv, ranked so you can see which misclassifications
    drive false positives.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import Config
from .dataset import build_dataloaders
from .model import build_model


def _load_model(cfg: Config, num_classes: int, device):
    checkpoint = torch.load(cfg.model_path, map_location=device)
    cfg.backbone = checkpoint.get("backbone", cfg.backbone)
    model = build_model(cfg, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def _collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def _plot_confusion_matrix(cm, class_names, out_path):
    """Render the confusion matrix to a PNG (matplotlib, no seaborn needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.6),
                                    max(5, len(class_names) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Crop Classifier - Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells only for small matrices to keep the plot readable.
    if len(class_names) <= 15:
        thresh = cm.max() / 2.0 if cm.max() else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _error_analysis(cm, class_names, out_path):
    """Rank off-diagonal confusions to spot false-positive drivers."""
    rows = []
    for i, true_cls in enumerate(class_names):
        row_total = cm[i].sum()
        for j, pred_cls in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "true_class": true_cls,
                    "predicted_as": pred_cls,
                    "count": int(cm[i, j]),
                    "pct_of_true_class": round(100.0 * cm[i, j] / max(row_total, 1), 2),
                })
    rows.sort(key=lambda r: r["count"], reverse=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["true_class", "predicted_as", "count", "pct_of_true_class"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def evaluate(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_dl, class_names = build_dataloaders(cfg)
    model = _load_model(cfg, len(class_names), device)

    y_true, y_pred = _collect_predictions(model, val_dl, device)

    # --- precision / recall / F1 ---
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = float((y_true == y_pred).mean())
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Macro precision: {macro_p:.4f} | recall: {macro_r:.4f} | F1: {macro_f1:.4f}")

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_path = os.path.join(cfg.artifacts_dir, "confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, cm_path)
    print(f"\nConfusion matrix saved -> {cm_path}")

    # --- error analysis ---
    ea_path = os.path.join(cfg.artifacts_dir, "error_analysis.csv")
    top_errors = _error_analysis(cm, class_names, ea_path)
    print(f"Error analysis saved  -> {ea_path}")
    if top_errors:
        print("\nTop confused pairs (drivers of false positives):")
        for r in top_errors[:5]:
            print(f"  {r['true_class']} -> {r['predicted_as']}: "
                  f"{r['count']} ({r['pct_of_true_class']}% of class)")

    # Persist headline metrics for the README / reports.
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }
    with open(os.path.join(cfg.artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Evaluate KrishiGPT crop classifier")
    p.add_argument("--data-dir")
    args = p.parse_args()
    cfg = Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    return cfg


if __name__ == "__main__":
    evaluate(_parse_args())
