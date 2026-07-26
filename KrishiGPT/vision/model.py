"""
vision/model.py
---------------
Transfer-learning model builder for crop-disease classification.

We start from an ImageNet-pretrained CNN backbone (ResNet-18 or MobileNetV2)
and replace its final layer with a fresh classifier head sized to the number
of crop classes. Optionally freezing the backbone trains only the head first
-- fast, data-efficient, and the reason transfer learning reaches high
validation accuracy on modest crop datasets.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .config import Config


def build_model(cfg: Config, num_classes: int) -> nn.Module:
    """Construct a transfer-learning classifier with ``num_classes`` outputs."""
    backbone = cfg.backbone.lower()

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        model = models.resnet18(weights=weights)
        if cfg.freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)  # new trainable head

    elif backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if cfg.freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported backbone: {cfg.backbone!r}")

    return model


def trainable_parameters(model: nn.Module):
    """Yield only the parameters that require gradients (for the optimizer)."""
    return (p for p in model.parameters() if p.requires_grad)
