"""
vision/predict.py
-----------------
Inference helper for the trained crop-disease classifier.

Used both from the CLI:
    python -m vision.predict path/to/leaf.jpg
and from the Flask app (the /predict-image endpoint) via ``predict_image``.
"""

from __future__ import annotations

import argparse
import functools

import torch
from PIL import Image

from .config import Config
from .dataset import build_transforms
from .model import build_model


@functools.lru_cache(maxsize=1)
def _load(cfg_key: str = "default"):
    """Load the checkpoint once and cache the (model, classes, transform, device)."""
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cfg.model_path, map_location=device)
    classes = checkpoint["classes"]
    cfg.backbone = checkpoint.get("backbone", cfg.backbone)

    model = build_model(cfg, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    _, eval_tf = build_transforms(cfg)
    return model, classes, eval_tf, device


def predict_image(image_path_or_file, top_k: int = 3):
    """Classify a crop image and return the top-k (label, probability) pairs.

    Accepts a file path or any file-like object PIL can open (e.g. a Flask
    upload stream).
    """
    model, classes, transform, device = _load()

    image = Image.open(image_path_or_file).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    k = min(top_k, len(classes))
    top_probs, top_idx = probs.topk(k)
    return [
        {"label": classes[i], "confidence": round(float(p), 4)}
        for p, i in zip(top_probs.tolist(), top_idx.tolist())
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Classify a crop leaf image")
    p.add_argument("image", help="path to an image file")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    for r in predict_image(args.image, top_k=args.top_k):
        print(f"{r['label']:<35} {r['confidence']*100:5.1f}%")
