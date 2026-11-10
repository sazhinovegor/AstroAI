"""Grad-CAM для визуального анализа решений EfficientNet-B0.

Тепловая карта показывает области, влияющие на выбранный логит модели.
Она не является доказательством физической значимости признака и не заменяет
проверку на контрольных изображениях.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from model_utils import SquarePadResize


def make_gradcam(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    class_index: int | None = None,
    alpha: float = 0.42,
) -> tuple[Image.Image, int, float]:
    """Возвращает overlay, индекс класса и оценку softmax для класса.

    Функция ожидает EfficientNet-подобную модель с `features`, `avgpool`
    и `classifier` (как torchvision.models.efficientnet_b0).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha должен находиться в диапазоне [0, 1]")

    model.eval()
    parameter = next(model.parameters())
    device = parameter.device
    x = transform(image).unsqueeze(0).to(device)

    # Вызов блоков вручную нужен, чтобы получить градиенты по последней
    # карте признаков, не используя устаревшие backward hooks.
    with torch.enable_grad():
        features = model.features(x)
        pooled = model.avgpool(features)
        logits = model.classifier(torch.flatten(pooled, 1))
        probs = torch.softmax(logits.float(), dim=1)
        predicted = int(probs.argmax(dim=1).item())
        selected = predicted if class_index is None else int(class_index)
        if not 0 <= selected < logits.shape[1]:
            raise ValueError("class_index выходит за диапазон классов модели")
        gradients = torch.autograd.grad(logits[0, selected], features, retain_graph=False)[0]

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().float().cpu().numpy()
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    base = SquarePadResize(x.shape[-1])(image)
    heat = Image.fromarray(np.uint8(np.clip(cam * 255, 0, 255)), mode="L")
    heat = ImageOps.colorize(heat, black="#160b39", mid="#d43d75", white="#fff36b", blackpoint=0, whitepoint=255)
    overlay = Image.blend(base, heat, alpha=alpha)
    return overlay, selected, float(probs[0, selected].detach().cpu().item())
