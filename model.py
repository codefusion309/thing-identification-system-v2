"""
model.py - MobileNetV2 Backbone & Embedding Extraction

Uses pretrained MobileNetV2 as a frozen feature extractor.
Produces 1280-dim embedding vectors used for nearest-neighbor classification.

No classifier head. No training. No retraining on corrections.
CPU only mode.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_device() -> torch.device:
    device = torch.device("cpu")
    print("[Device] Using CPU.")
    return device


def load_backbone() -> nn.Module:
    """
    Load pretrained MobileNetV2 as a frozen feature extractor.

    On first run:  downloads ~14 MB weights from PyTorch servers.
    After that:    loads instantly from local cache (~/.cache/torch/).
    Offline use:   pre-cache the weights before going offline (see README).
    """
    model = models.mobilenet_v2(weights=None)  # don't auto-download
    state = torch.load("mobilenet_v2_weights.pth", map_location="cpu")
    model.load_state_dict(state)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    print("[Model] MobileNetV2 backbone loaded (pretrained, frozen).")
    return model


def get_embedding(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract a 1280-dim feature vector from the MobileNetV2 backbone.

    Args:
        model:        MobileNetV2 loaded via load_backbone().
        image_tensor: Preprocessed image tensor, shape (1, 3, 224, 224).

    Returns:
        Tensor of shape (1, 1280).
    """
    with torch.no_grad():
        features  = model.features(image_tensor)                          # (1, 1280, 7, 7)
        features  = nn.functional.adaptive_avg_pool2d(features, (1, 1))  # (1, 1280, 1, 1)
        embedding = features.flatten(1)                                    # (1, 1280)
    return embedding
