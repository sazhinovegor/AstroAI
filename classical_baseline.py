"""Классический ML-baseline для сравнения с EfficientNet-B0.

Признаки: нормированные RGB-гистограммы, миниатюра яркости и цветовые
статистики. Это прозрачный контрольный метод, а не конкурент современным
нейросетевым архитектурам. Параметры не подбираются по test-набору.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import ImageFolder

from model_utils import SquarePadResize


def extract_features(image: Image.Image) -> np.ndarray:
    """Извлекает воспроизводимый вектор признаков фиксированной длины."""
    image = SquarePadResize(64)(image)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    features: list[float] = []

    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        features.extend(hist.tolist())

    gray = np.asarray(image.convert("L").resize((12, 12), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    features.extend(gray.flatten().tolist())

    for channel in range(3):
        values = rgb[:, :, channel]
        features.extend([float(values.mean()), float(values.std())])

    vector = np.asarray(features, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Извлечённый вектор содержит NaN или Inf")
    return vector


def load_split(folder: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    dataset = ImageFolder(str(folder))
    if not dataset.samples:
        raise SystemExit(f"Раздел не содержит изображений: {folder}")
    vectors, labels, paths = [], [], []
    for path, label in dataset.samples:
        try:
            with Image.open(path) as source:
                vectors.append(extract_features(source.copy()))
        except Exception as exc:
            raise SystemExit(f"Не удалось обработать {path}: {exc}") from exc
        labels.append(label)
        paths.append(Path(path).relative_to(folder.parent).as_posix())
    return np.vstack(vectors), np.asarray(labels, dtype=int), paths, list(dataset.classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Классический baseline AstroVision AI")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="research_runs/classical_baseline")
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    train_x, train_y, _, classes = load_split(root / "train")
    test_x, test_y, test_paths, test_classes = load_split(root / "test")
    if classes != test_classes or len(classes) != 3:
        raise SystemExit(f"Ожидаются одинаковые три класса (galaxies, nebulae, other) в train и test: {classes} / {test_classes}")
    if set(np.unique(train_y)) != {0, 1} or set(np.unique(test_y)) != {0, 1}:
        raise SystemExit("В train и test должны присутствовать оба класса")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=args.seed),
    )
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)
    predictions = probabilities.argmax(axis=1)

    metrics = {
        "baseline": "RGB histogram + grayscale thumbnail + logistic regression",
        "features": int(train_x.shape[1]),
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "classes": classes,
        "seed": args.seed,
        "accuracy": float(accuracy_score(test_y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "precision_macro": float(precision_score(test_y, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(test_y, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(test_y, predictions, average="macro", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(test_y, predictions)),
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=list(range(len(classes)))).tolist(),
        "classification_report": classification_report(
            test_y, predictions, labels=list(range(len(classes))), target_names=classes,
            output_dict=True, zero_division=0,
        ),
    }
    if len(np.unique(test_y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(test_y, probabilities[:, 1]))

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_class", "predicted_class", *[f"prob_{name}" for name in classes]])
        for path, truth, prediction, probs in zip(test_paths, test_y, predictions, probabilities):
            writer.writerow([path, classes[int(truth)], classes[int(prediction)], *map(float, probs)])

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nРезультаты baseline сохранены в {output.resolve()}")


if __name__ == "__main__":
    main()
