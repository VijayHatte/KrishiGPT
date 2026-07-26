"""
vision/config.py
----------------
Central configuration for the crop-disease image classifier.

Dataset layout expected (torchvision ImageFolder convention):

    data/crop_dataset/
        train/
            Tomato___healthy/         img001.jpg ...
            Tomato___Late_blight/     ...
            Potato___Early_blight/    ...
            ...
        val/
            Tomato___healthy/         ...
            ...

Any ImageFolder-style crop dataset works (e.g. the open PlantVillage dataset).
Class names are inferred from the sub-folder names at load time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# Resolve paths relative to this file so scripts work from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)


@dataclass
class Config:
    # --- data ---
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "crop_dataset")
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 224                      # matches ImageNet-pretrained backbones

    # ImageNet normalisation stats (transfer learning from ImageNet weights).
    norm_mean: tuple = (0.485, 0.456, 0.406)
    norm_std: tuple = (0.229, 0.224, 0.225)

    # --- model ---
    backbone: str = "resnet18"                 # resnet18 | mobilenet_v2
    pretrained: bool = True                    # ImageNet weights for transfer learning
    freeze_backbone: bool = True               # train the classifier head first

    # --- training ---
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    balance_classes: bool = True               # weighted sampler for imbalance
    seed: int = 42

    # --- artefacts ---
    artifacts_dir: str = os.path.join(PROJECT_ROOT, "artifacts")
    model_path: str = field(default="")        # filled in __post_init__
    labels_path: str = field(default="")

    def __post_init__(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        if not self.model_path:
            self.model_path = os.path.join(self.artifacts_dir, "crop_model.pt")
        if not self.labels_path:
            self.labels_path = os.path.join(self.artifacts_dir, "labels.json")
