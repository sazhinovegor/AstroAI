"""Общие преобразования изображений AstroVision AI.

Геометрическая предобработка едина для обучения, оценки и инференса.
EXIF Orientation применяется до изменения размера, чтобы снимки с камер
и мобильных устройств не попадали в модель повернутыми неверно.
"""
from __future__ import annotations

from PIL import Image, ImageOps
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SquarePadResize:
    """Вписывает изображение в квадрат без искажения пропорций сторон."""

    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)) -> None:
        if size < 32:
            raise ValueError("Размер изображения должен быть не меньше 32 px")
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("Ожидается объект PIL.Image.Image")
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.pad(
            image,
            (self.size, self.size),
            method=Image.Resampling.LANCZOS,
            color=self.fill,
            centering=(0.5, 0.5),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, fill={self.fill})"


def build_eval_transform(img_size: int = 224):
    """Детерминированная предобработка для validation, test и инференса."""
    return transforms.Compose([
        SquarePadResize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
