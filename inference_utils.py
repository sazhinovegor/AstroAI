"""Единый безопасный инференс для одиночных и пакетных астрофотографий."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import torch
from PIL import Image, ImageOps

from model_utils import build_eval_transform


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    status: str
    inference_ms: float
    rejection_reason: str = ""


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    classes: list[str],
    img_size: int = 224,
    threshold: float = 0.65,
    tta: bool = False,
) -> Prediction:
    """Классифицирует изображение; класс other и низкая уверенность означают отказ."""
    if not classes or len(classes) < 2:
        raise ValueError("Для классификации требуется не менее двух классов")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold должен находиться в диапазоне [0, 1]")

    image = ImageOps.exif_transpose(image).convert("RGB")
    x = build_eval_transform(img_size)(image).unsqueeze(0)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    x = x.to(device)

    model.eval()
    started = perf_counter()
    with torch.inference_mode():
        logits = model(x).float()
        probs = torch.softmax(logits, dim=1)
        if tta:
            mirrored = model(torch.flip(x, dims=[3])).float()
            probs = (probs + torch.softmax(mirrored, dim=1)) / 2
    elapsed = (perf_counter() - started) * 1000

    if probs.ndim != 2 or probs.shape != (1, len(classes)):
        raise ValueError("Размер выхода модели не совпадает с количеством классов")
    values = probs[0].detach().cpu().tolist()
    index = int(torch.argmax(probs, dim=1).item())
    confidence = float(values[index])
    predicted = classes[index]
    if predicted == "other":
        status, reason = "not_supported", "Изображение не относится к галактикам или туманностям."
    elif confidence < threshold:
        status, reason = "not_supported", "Не удалось уверенно отнести изображение к галактикам или туманностям."
    else:
        status, reason = "predicted", ""
    return Prediction(
        predicted_class=predicted, confidence=confidence,
        probabilities={name: float(value) for name, value in zip(classes, values)},
        status=status, inference_ms=elapsed, rejection_reason=reason,
    )


def predict_many(
    model: torch.nn.Module,
    images: Iterable[tuple[str, Image.Image]],
    classes: list[str],
    img_size: int = 224,
    threshold: float = 0.65,
    tta: bool = False,
) -> list[dict]:
    """Пакетная обработка с локальной изоляцией ошибок отдельных файлов."""
    results = []
    for filename, image in images:
        try:
            result = predict_image(model, image, classes, img_size, threshold, tta)
            results.append({
                "filename": filename, "predicted_class": result.predicted_class,
                "confidence": result.confidence, "status": result.status, "rejection_reason": result.rejection_reason,
                "inference_ms": result.inference_ms,
                **{f"prob_{key}": value for key, value in result.probabilities.items()},
                "error": "",
            })
        except Exception as exc:  # одна плохая картинка не прерывает весь пакет
            results.append({"filename": filename, "predicted_class": "", "confidence": None,
                            "status": "error", "inference_ms": None, "error": str(exc)})
    return results
