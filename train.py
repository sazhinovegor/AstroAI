"""AstroVision AI v0.5 — воспроизводимое обучение классификатора астрофотографий.

Классы: galaxies и nebulae. Модель: EfficientNet-B0 + transfer learning.
Важные принципы: фиксированный seed, контроль утечек данных, best checkpoint
по validation loss и итоговая оценка только на отдельном test-наборе.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from model_utils import SquarePadResize, build_eval_transform, MEAN, STD

EXPECTED_CLASSES = ["galaxies", "nebulae"]


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Воспроизводимость предпочтительнее небольшой прибавки скорости.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms(img_size: int = 224, augment: bool = True):
    # Letterbox сохраняет пропорции сторон: вытягивание астрофотографии в квадрат
    # может менять морфологические признаки объекта.
    train_ops = [SquarePadResize(img_size)]
    if augment:
        # Для этой задачи ориентация объекта не является целевым признаком.
        # Цветовые сдвиги намеренно небольшие, чтобы не стирать цветовые различия.
        train_ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20, fill=(0, 0, 0)),
            transforms.RandomResizedCrop(img_size, scale=(0.88, 1.0)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06),
        ]
    train_ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(train_ops), build_eval_transform(img_size)


def write_json(path: str | Path | None, payload: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_split_leakage(split_dirs: dict[str, Path]) -> None:
    """Останавливает обучение при идентичных файлах между разными разделами."""
    seen: dict[str, tuple[str, Path]] = {}
    duplicates = []
    for split, folder in split_dirs.items():
        if not folder.exists():
            continue
        for file in folder.rglob("*"):
            if not file.is_file():
                continue
            try:
                digest = sha256_file(file)
            except OSError as exc:
                raise SystemExit(f"Не удалось прочитать файл {file}: {exc}")
            if digest in seen and seen[digest][0] != split:
                old_split, old_path = seen[digest]
                duplicates.append((old_split, str(old_path), split, str(file)))
            else:
                seen[digest] = (split, file)
    if duplicates:
        details = "\n".join(
            f"{a}: {b} == {c}: {d}" for a, b, c, d in duplicates[:20]
        )
        raise SystemExit(
            "Обнаружены одинаковые файлы между train/val/test. Это может завысить "
            "качество на тесте. Удалите дубликаты или исправьте разделение:\n" + details
        )


def check_metadata_integrity(root: Path, required: bool = False,
                             require_provenance: bool = False) -> list[str]:
    """Проверяет связь изображений с object_id и источником в metadata.csv.

    Возвращает предупреждения. Ошибки структуры/утечки останавливают обучение.
    """
    metadata_path = root / "metadata.csv"
    if not metadata_path.exists():
        message = (
            "metadata.csv не найден: точные копии проверены, но разделение по "
            "астрономическим объектам и происхождение снимков не подтверждены."
        )
        if required:
            raise SystemExit(message + " Для конференционного запуска добавьте metadata.csv.")
        return [message]

    required_columns = {"path", "object_id"}
    rows_by_path: dict[str, dict[str, str]] = {}
    object_splits: dict[str, set[str]] = {}
    object_classes: dict[str, set[str]] = {}
    warnings: list[str] = []
    missing_provenance: list[str] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not required_columns.issubset(fields):
            raise SystemExit("metadata.csv должен содержать колонки path и object_id")
        for line_number, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip().replace("\\", "/")
            object_id = (row.get("object_id") or "").strip()
            if not raw_path or not object_id:
                raise SystemExit(f"metadata.csv: пустой path или object_id в строке {line_number}")
            rel = Path(raw_path)
            if rel.is_absolute() or ".." in rel.parts:
                raise SystemExit(f"metadata.csv: небезопасный относительный путь в строке {line_number}: {raw_path}")
            normalized = rel.as_posix()
            if normalized in rows_by_path:
                raise SystemExit(f"metadata.csv: повторяется путь {normalized}")
            parts = rel.parts
            if len(parts) < 3 or parts[0] not in {"train", "val", "test"} or parts[1] not in EXPECTED_CLASSES:
                raise SystemExit(
                    f"metadata.csv: путь должен иметь вид split/class/file: {normalized}"
                )
            file_path = root / rel
            if not file_path.is_file():
                raise SystemExit(f"metadata.csv ссылается на отсутствующий файл: {normalized}")
            rows_by_path[normalized] = row
            object_splits.setdefault(object_id, set()).add(parts[0])
            object_classes.setdefault(object_id, set()).add(parts[1])
            source = (row.get("source") or "").strip()
            license_name = (row.get("license") or "").strip()
            if not source or not license_name:
                missing_provenance.append(normalized)
            if require_provenance and (not source or not license_name):
                raise SystemExit(
                    f"Для конференционного запуска заполните source и license: {normalized}"
                )

    mislabeled = {obj: sorted(classes) for obj, classes in object_classes.items() if len(classes) > 1}
    if mislabeled:
        examples = "; ".join(f"{obj}: {classes}" for obj, classes in list(mislabeled.items())[:10])
        raise SystemExit("Один object_id отмечен разными классами: " + examples)
    leaked = {obj: sorted(splits) for obj, splits in object_splits.items() if len(splits) > 1}
    if leaked:
        examples = "; ".join(f"{obj}: {splits}" for obj, splits in list(leaked.items())[:10])
        raise SystemExit("Утечка по object_id между train/val/test: " + examples)

    image_paths = set()
    for split in ("train", "val", "test"):
        for class_name in EXPECTED_CLASSES:
            class_dir = root / split / class_name
            if class_dir.exists():
                image_paths.update(
                    p.relative_to(root).as_posix() for p in class_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                )
    missing_rows = sorted(image_paths - set(rows_by_path))
    stale_rows = sorted(set(rows_by_path) - image_paths)
    if missing_rows:
        msg = f"В metadata.csv отсутствуют {len(missing_rows)} файлов; пример: {missing_rows[:3]}"
        if required:
            raise SystemExit(msg)
        warnings.append(msg)
    if stale_rows:
        raise SystemExit(f"В metadata.csv указаны файлы, отсутствующие в структуре датасета: {stale_rows[:5]}")
    if missing_provenance:
        warnings.append(
            f"В metadata.csv не заполнены source/license у {len(missing_provenance)} изображений; "
            "для конференционного запуска используйте --require_provenance."
        )
    return warnings


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              use_amp=False, channels_last=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    start = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp else nullcontext()
        )
        with amp_context:
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return total_loss / max(total, 1), correct / max(total, 1), elapsed


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    return float(ece)


def stratified_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                            n_bootstrap: int = 1000, seed: int = 2026,
                            object_ids: list[str] | None = None) -> dict:
    """Стратифицированный bootstrap по объектам; без metadata — по изображениям.

    Если переданы object_ids, целиком пересэмплируются астрономические объекты,
    а не отдельные кадры. Это уменьшает псевдорепликацию при нескольких снимках
    одного объекта в test-наборе.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    values = {"accuracy": [], "balanced_accuracy": [], "f1_macro": []}

    if object_ids is not None and len(object_ids) == len(y_true):
        group_ids = np.asarray(object_ids, dtype=str)
        class_groups = []
        for cls in classes:
            indices = np.flatnonzero(y_true == cls)
            unique_groups = np.unique(group_ids[indices])
            # Каждая группа должна принадлежать ровно одному классу.
            for group_id in unique_groups:
                if len(np.unique(y_true[group_ids == group_id])) != 1:
                    raise ValueError(f"object_id {group_id!r} встречается в нескольких классах")
            class_groups.append(unique_groups)
        bootstrap_unit = "object_id"
    else:
        class_groups = [np.flatnonzero(y_true == cls) for cls in classes]
        group_ids = None
        bootstrap_unit = "image"

    for _ in range(n_bootstrap):
        sampled_indices = []
        for candidates in class_groups:
            sampled = rng.choice(candidates, size=len(candidates), replace=True)
            if group_ids is None:
                sampled_indices.extend(sampled.tolist())
            else:
                for group_id in sampled:
                    sampled_indices.extend(np.flatnonzero(group_ids == group_id).tolist())
        indices = np.asarray(sampled_indices, dtype=int)
        yt, yp = y_true[indices], y_pred[indices]
        values["accuracy"].append(accuracy_score(yt, yp))
        values["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        values["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
    result = {
        key: [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        for key, arr in values.items()
    }
    result["bootstrap_unit"] = bootstrap_unit
    result["n_bootstrap"] = n_bootstrap
    return result


def read_object_ids(data_root: Path, sample_paths: list[str]) -> list[str] | None:
    """Возвращает object_id в порядке ImageFolder; None при отсутствии полного metadata."""
    metadata_path = data_root / "metadata.csv"
    if not metadata_path.is_file():
        return None
    lookup = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = (row.get("path") or "").strip().replace("\\", "/")
            object_id = (row.get("object_id") or "").strip()
            if path and object_id:
                lookup[Path(path).as_posix()] = object_id
    values = [lookup.get(Path(path).as_posix()) for path in sample_paths]
    return [str(value) for value in values] if all(values) else None


def evaluate_test(model, loader, classes, device, use_amp, out_prefix: Path, seed: int,
                  sample_paths: list[str] | None = None,
                  sample_object_ids: list[str] | None = None):
    model.eval()
    ys, preds, probs_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=device.type == "cuda")
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp else nullcontext()
            )
            with amp_context:
                logits = model(x)
                probs = torch.softmax(logits.float(), dim=1)
            probs_all.extend(probs.cpu().numpy().tolist())
            preds.extend(probs.argmax(1).cpu().tolist())
            ys.extend(y.tolist())
    y_true = np.asarray(ys, dtype=int)
    y_pred = np.asarray(preds, dtype=int)
    probabilities = np.asarray(probs_all, dtype=float)
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)[0]),
        "recall_macro": float(precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)[1]),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "brier_score": float(np.mean(np.sum(
            (probabilities - np.eye(len(classes))[y_true]) ** 2, axis=1
        ))),
        "ece_15_bins": expected_calibration_error(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            classes[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            } for i in labels
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=classes,
            output_dict=True, zero_division=0
        ),
        "bootstrap_95ci": stratified_bootstrap_ci(
            y_true, y_pred, seed=seed, object_ids=sample_object_ids
        ),
    }
    if len(classes) == 2 and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(out_prefix) + ".metrics.json", metrics)
    with (Path(str(out_prefix) + ".predictions.csv")).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_index", "true_class", "predicted_index", "predicted_class",
                         "confidence", *[f"prob_{c}" for c in classes]])
        for i, (yt, yp, prob) in enumerate(zip(y_true, y_pred, probabilities)):
            sample_path = sample_paths[i] if sample_paths and i < len(sample_paths) else ""
            writer.writerow([sample_path, int(yt), classes[yt], int(yp), classes[yp],
                             float(prob.max()), *prob.tolist()])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="AstroVision AI v0.5 research training")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.65,
                         help="Порог низкой уверенности для интерфейса; не является OOD-детектором")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--class_weighted_loss", action="store_true",
                         help="Компенсировать дисбаланс классов весами loss")
    parser.add_argument("--from_scratch", action="store_true",
                         help="Инициализировать EfficientNet-B0 случайно, без весов ImageNet (контрольный эксперимент)")
    parser.add_argument("--require_metadata", action="store_true",
                         help="Остановиться, если metadata.csv не содержит object_id для всех снимков")
    parser.add_argument("--require_provenance", action="store_true",
                         help="Дополнительно требовать source и license для каждого снимка")
    parser.add_argument("--allow_missing_test", action="store_true",
                         help="Только для проверки кода; для конференционного результата не использовать")
    parser.add_argument("--progress_file", default=None)
    parser.add_argument("--stop_file", default=None)
    parser.add_argument("--out", default="models/astrovision_model.pth")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.img_size < 32:
        raise SystemExit("epochs, batch_size должны быть положительными, img_size >= 32")
    if not 0 <= args.freeze_epochs <= args.epochs:
        raise SystemExit("freeze_epochs должен быть от 0 до epochs")

    set_reproducibility(args.seed)
    root = Path(args.data_dir)
    split_dirs = {name: root / name for name in ("train", "val", "test")}
    for name in ("train", "val"):
        if not split_dirs[name].exists():
            raise SystemExit(f"Не найдена папка {split_dirs[name]}")
    if not split_dirs["test"].exists() and not args.allow_missing_test:
        raise SystemExit("Для исследовательского запуска обязательна папка dataset/test. "
                         "--allow_missing_test допускается только для проверки кода.")
    check_split_leakage(split_dirs)
    for warning in check_metadata_integrity(
        root, required=args.require_metadata, require_provenance=args.require_provenance
    ):
        print("ПРЕДУПРЕЖДЕНИЕ:", warning)

    train_tf, eval_tf = build_transforms(args.img_size, augment=not args.no_aug)
    train_ds = datasets.ImageFolder(split_dirs["train"], transform=train_tf)
    val_ds = datasets.ImageFolder(split_dirs["val"], transform=eval_tf)
    if train_ds.classes != EXPECTED_CLASSES:
        raise SystemExit(f"Ожидаются два класса {EXPECTED_CLASSES}; найдены {train_ds.classes}")
    if val_ds.classes != train_ds.classes:
        raise SystemExit(f"Классы train {train_ds.classes} не совпадают с val {val_ds.classes}")
    if not train_ds.samples or not val_ds.samples:
        raise SystemExit("Папки train и val должны содержать изображения")
    expected_targets = set(range(len(train_ds.classes)))
    if set(train_ds.targets) != expected_targets:
        raise SystemExit("В train должны присутствовать изображения обоих классов")
    if set(val_ds.targets) != expected_targets:
        raise SystemExit("В val должны присутствовать изображения обоих классов")

    test_ds = None
    if split_dirs["test"].exists():
        test_ds = datasets.ImageFolder(split_dirs["test"], transform=eval_tf)
        if test_ds.classes != train_ds.classes:
            raise SystemExit("Классы test не совпадают с train")
        if not test_ds.samples:
            raise SystemExit("Папка test существует, но не содержит изображений")
        if set(test_ds.targets) != set(range(len(train_ds.classes))):
            raise SystemExit("В test должны присутствовать оба класса")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    channels_last = device.type == "cuda"
    print(f"Устройство: {device}; GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'нет'}")
    print(f"AMP: {use_amp}; seed: {args.seed}; классы: {train_ds.classes}")
    print(f"Размеры: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds) if test_ds else 0}")

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs) if test_ds else None

    if args.from_scratch and args.freeze_epochs > 0:
        raise SystemExit("Для --from_scratch используйте --freeze_epochs 0: случайно инициализированную backbone нельзя замораживать.")
    weights = None if args.from_scratch else models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_ds.classes))
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)

    class_weights = None
    if args.class_weighted_loss:
        counts = np.bincount(np.asarray(train_ds.targets), minlength=len(train_ds.classes))
        if np.any(counts == 0):
            raise SystemExit("Каждый класс должен содержать train-изображения")
        class_weights = torch.tensor(len(train_ds) / (len(counts) * counts),
                                     dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    head_params = list(model.classifier.parameters())
    head_ids = {id(param) for param in head_params}
    backbone_params = [param for param in model.parameters() if id(param) not in head_ids]
    for param in backbone_params:
        param.requires_grad_(args.freeze_epochs == 0)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    def schedule_factor(epoch_index: int) -> float:
        if args.epochs <= 2 or epoch_index == 0:
            return 1.0
        progress = (epoch_index - 1) / max(args.epochs - 2, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history, wait = [], 0
    status, stopped_message = "done", ""
    payload = {"status": "running", "epochs": args.epochs, "epoch": 0,
               "classes": train_ds.classes, "device": str(device), "amp": use_amp,
               "seed": args.seed}
    write_json(args.progress_file, payload)

    for epoch in range(args.epochs):
        is_frozen = epoch < args.freeze_epochs
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            for param in backbone_params:
                param.requires_grad_(True)
            print(f"Эпоха {epoch + 1}: backbone разморожен")
        factor = schedule_factor(epoch)
        # LR выставляется до эпохи. Так первая размороженная эпоха получает
        # ненулевой LR — исправление ошибки прежнего LambdaLR-сценария.
        optimizer.param_groups[0]["lr"] = 0.0 if is_frozen else args.lr * 0.1 * factor
        optimizer.param_groups[1]["lr"] = args.lr * factor

        train_loss, train_acc, train_seconds = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
            use_amp=use_amp, channels_last=channels_last)
        val_loss, val_acc, _ = run_epoch(
            model, val_loader, criterion, device,
            use_amp=use_amp, channels_last=channels_last)
        images_per_sec = len(train_ds) / max(train_seconds, 1e-9)
        print(f"Эпоха {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}, "
              f"train_acc={train_acc:.3f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_acc:.3f}, backbone_lr={optimizer.param_groups[0]['lr']:.2e}, "
              f"head_lr={optimizer.param_groups[1]['lr']:.2e}, "
              f"{images_per_sec:.1f} изображений/с")
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 6), "val_acc": round(val_acc, 6),
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "epoch_sec": round(train_seconds, 3),
            "images_per_sec": round(images_per_sec, 2),
        })
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        payload.update({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val_loss, "train_acc": train_acc,
                        "val_acc": val_acc, "img_per_sec": images_per_sec,
                        "best_val_loss": best_val_loss, "history": history})
        write_json(args.progress_file, payload)

        if args.stop_file and Path(args.stop_file).exists():
            Path(args.stop_file).unlink(missing_ok=True)
            status, stopped_message = "stopped", "Остановлено пользователем после завершения эпохи"
            break
        if args.patience > 0 and wait >= args.patience:
            status, stopped_message = "early_stopped", f"Early stopping после {epoch + 1} эпох"
            print(stopped_message)
            break

    model.load_state_dict(best_state)
    metrics = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_prefix = out.with_suffix("")
    if test_loader is not None:
        test_paths = [str(Path(path).relative_to(root)) for path, _ in test_ds.samples]
        test_object_ids = read_object_ids(root, test_paths)
        metrics = evaluate_test(model, test_loader, train_ds.classes, device,
                                use_amp, report_prefix, args.seed,
                                sample_paths=test_paths, sample_object_ids=test_object_ids)
        print("Итоговые test-метрики:", json.dumps(
            {k: v for k, v in metrics.items() if k not in ("per_class", "classification_report", "confusion_matrix")},
            ensure_ascii=False, indent=2))

    experiment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "arguments": vars(args),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
        "dataset_counts": {
            "train": len(train_ds), "val": len(val_ds),
            "test": len(test_ds) if test_ds else 0,
        },
        "classes": train_ds.classes,
        "architecture": "EfficientNet-B0",
        "pretrained_imagenet": not args.from_scratch,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "history": history,
        "status": status,
        "message": stopped_message,
    }
    write_json(str(report_prefix) + ".experiment.json", experiment)
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classes": train_ds.classes,
        "threshold": args.threshold,
        "metrics": metrics,
        "history": history,
        "img_size": args.img_size,
        "arch": "efficientnet_b0",
        "pretrained_imagenet": not args.from_scratch,
        "seed": args.seed,
        "best_val_loss": best_val_loss,
        "experiment": experiment,
    }, out)
    payload.update({"status": status, "metrics": metrics,
                    "model_path": str(out), "message": stopped_message})
    write_json(args.progress_file, payload)
    print(f"Модель сохранена: {out}")
    print(f"Отчёты: {report_prefix}.metrics.json, {report_prefix}.predictions.csv, {report_prefix}.experiment.json")


if __name__ == "__main__":
    main()
