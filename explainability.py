"""Архитектурно-независимый Grad-CAM для моделей torchvision."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from model_utils import SquarePadResize


def make_gradcam(model: torch.nn.Module, image: Image.Image, transform,
                 class_index: int | None = None, alpha: float = 0.42):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha должен находиться в диапазоне [0, 1]")
    model.eval()
    parameter = next(model.parameters())
    device = parameter.device
    x = transform(image).unsqueeze(0).to(device)
    # Последний свёрточный слой подходит для EfficientNet, ResNet и ConvNeXt.
    target_layer = next((m for m in reversed(list(model.modules()))
                         if isinstance(m, torch.nn.Conv2d)), None)
    if target_layer is None:
        raise ValueError("В архитектуре не найден свёрточный слой для Grad-CAM")
    captured = {}
    def capture(_module, _inputs, output):
        captured["features"] = output
    handle = target_layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = model(x)
            features = captured.get("features")
            if features is None or not features.requires_grad:
                raise RuntimeError("Не удалось получить признаки для Grad-CAM")
            probs = torch.softmax(logits.float(), dim=1)
            predicted = int(probs.argmax(dim=1).item())
            selected = predicted if class_index is None else int(class_index)
            if not 0 <= selected < logits.shape[1]:
                raise ValueError("class_index выходит за диапазон классов модели")
            gradients = torch.autograd.grad(logits[0, selected], features)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().float().cpu().numpy()
        lo, hi = float(cam.min()), float(cam.max())
        cam = (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam, dtype=np.float32)
        base = SquarePadResize(x.shape[-1])(image)
        heat = Image.fromarray(np.uint8(np.clip(cam * 255, 0, 255)), mode="L")
        heat = ImageOps.colorize(heat, black="#160b39", mid="#d43d75", white="#fff36b")
        overlay = Image.blend(base, heat, alpha=alpha)
        return overlay, selected, float(probs[0, selected].detach().cpu().item())
    finally:
        handle.remove()
