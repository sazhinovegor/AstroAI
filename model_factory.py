"""Единая фабрика архитектур для обучения и инференса AstroVision AI."""
from __future__ import annotations
import torch.nn as nn
from torchvision import models

ARCHITECTURES = ("efficientnet_b0", "resnet50", "convnext_tiny")


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    if num_classes < 2:
        raise ValueError("Модель должна иметь не менее двух классов")
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Неизвестная архитектура: {arch}. Доступно: {', '.join(ARCHITECTURES)}")
    return model


def classifier_parameters(model, arch: str):
    if arch == "resnet50":
        return list(model.fc.parameters())
    if arch == "efficientnet_b0":
        return list(model.classifier.parameters())
    if arch == "convnext_tiny":
        return list(model.classifier.parameters())
    raise ValueError(f"Неизвестная архитектура: {arch}")
