"""Быстрые регрессионные тесты ключевых инвариантов AstroVision AI."""
from pathlib import Path

import numpy as np
from PIL import Image

from model_utils import SquarePadResize, build_eval_transform
from prepare_dataset import choose_grouped_split
from train import expected_calibration_error


def test_square_pad_resize_preserves_aspect_ratio_and_size():
    source = Image.new("RGB", (320, 160), (20, 40, 60))
    result = SquarePadResize(224)(source)
    assert result.size == (224, 224)
    # Вписывание в квадрат должно оставлять полосы, а не растягивать исходник.
    assert result.getpixel((112, 0)) == (0, 0, 0)
    assert result.getpixel((112, 112)) == (20, 40, 60)


def test_eval_transform_is_deterministic_shape_and_finite():
    image = Image.new("RGB", (80, 120), (120, 90, 30))
    transform = build_eval_transform(96)
    first, second = transform(image), transform(image)
    assert first.shape == (3, 96, 96)
    assert np.isfinite(first.numpy()).all()
    assert np.array_equal(first.numpy(), second.numpy())


def test_grouped_split_has_no_object_leakage_and_all_classes():
    labels, groups = [], []
    for cls in ("galaxies", "nebulae", "other"):
        for group_no in range(14):
            # Несколько кадров одного объекта проверяют именно групповой split.
            for _ in range(2):
                labels.append(cls)
                groups.append(f"{cls}_{group_no}")
    splits, details = choose_grouped_split(labels, groups, seed=17, attempts=3)
    group_sets = {name: {groups[i] for i in idx} for name, idx in splits.items()}
    assert not (group_sets["train"] & group_sets["val"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["val"] & group_sets["test"])
    for indices in splits.values():
        assert {labels[i] for i in indices} == {"galaxies", "nebulae", "other"}
    assert details["seed"] == 17


def test_grouped_split_rejects_mixed_class_object():
    try:
        choose_grouped_split(["galaxies", "nebulae", "other"], ["same", "same", "other_obj"])
    except ValueError as exc:
        assert "нескольких классах" in str(exc)
    else:
        raise AssertionError("Ожидалась ошибка для object_id с двумя классами")


def test_ece_is_zero_for_perfect_confident_predictions():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])
    assert abs(expected_calibration_error(y, p) - 0.01) < 1e-9


def test_predict_image_returns_class_probabilities_and_abstention():
    import torch
    from inference_utils import predict_image

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.tensor([[0.0, 0.0]], device=x.device) + self.anchor

    image = Image.new("RGB", (80, 60), (20, 40, 60))
    result = predict_image(FixedModel(), image, ["galaxies", "nebulae"], threshold=0.6)
    assert result.predicted_class == "galaxies"
    assert result.status == "not_supported"
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    assert result.inference_ms >= 0


def test_predict_many_isolates_bad_image():
    import torch
    from inference_utils import predict_many

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.tensor([[2.0, 0.0]], device=x.device) + self.anchor

    model = FixedModel()
    good = Image.new("RGB", (64, 64), (10, 20, 30))
    results = predict_many(model, [("good.png", good), ("broken.png", None)],
                           ["galaxies", "nebulae"])
    assert results[0]["status"] == "predicted"
    assert results[1]["status"] == "error"
    assert results[1]["filename"] == "broken.png"


def test_predict_image_rejects_other_class():
    import torch
    from inference_utils import predict_image

    class OtherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.tensor([[0.0, 0.0, 4.0]], device=x.device) + self.anchor

    image = Image.new("RGB", (80, 60), (20, 40, 60))
    result = predict_image(OtherModel(), image, ["galaxies", "nebulae", "other"])
    assert result.predicted_class == "other"
    assert result.status == "not_supported"
    assert "не относится" in result.rejection_reason
