"""Проверка датасета AstroVision AI перед обучением и научной оценкой."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SPLITS = ("train", "val", "test")
CLASSES = ("galaxies", "nebulae", "other")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Аудит датасета AstroVision AI")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--metadata", default=None, help="Путь к metadata.csv")
    parser.add_argument("--out", default="reports/dataset_audit.json")
    parser.add_argument("--require_metadata", action="store_true",
                        help="Считать отсутствие object_id/записей для снимков ошибкой")
    parser.add_argument("--require_provenance", action="store_true",
                        help="Требовать непустые source и license для каждого изображения")
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    report: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(root), "classes": list(CLASSES),
        "splits": {}, "errors": [], "warnings": [],
        "cross_split_exact_duplicates": [], "object_group_leakage": [],
        "metadata": {"rows": 0, "missing_images": [], "unlisted_images": [],
                     "duplicate_paths": [], "missing_provenance": []},
    }

    hashes: dict[str, tuple[str, str]] = {}
    split_counts: dict[str, Counter] = {}
    all_images: dict[str, str] = {}

    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            report["errors"].append(f"Не найдена папка {split_root}")
            continue
        counts = Counter()
        dimensions = Counter()
        invalid = []
        for class_name in CLASSES:
            class_root = split_root / class_name
            if not class_root.is_dir():
                report["errors"].append(f"Не найдена папка класса {class_root}")
                continue
            files = image_files(class_root)
            for file_path in files:
                rel = file_path.relative_to(root).as_posix()
                all_images[rel] = split
                counts[class_name] += 1
                try:
                    with Image.open(file_path) as image:
                        image.verify()
                    with Image.open(file_path) as image:
                        dimensions[f"{image.width}x{image.height}"] += 1
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    invalid.append({"path": rel, "error": str(exc)})
                    continue
                digest = sha256(file_path)
                if digest in hashes and hashes[digest][0] != split:
                    report["cross_split_exact_duplicates"].append({
                        "first": hashes[digest][1], "second": rel,
                    })
                else:
                    hashes[digest] = (split, rel)

        split_counts[split] = counts
        report["splits"][split] = {
            "counts": dict(counts), "total": int(sum(counts.values())),
            "image_dimensions": dict(dimensions), "invalid_images": invalid,
        }
        if invalid:
            report["errors"].append(f"{split}: повреждённых или нечитаемых изображений: {len(invalid)}")
        if any(counts[name] == 0 for name in CLASSES):
            report["errors"].append(f"{split}: один или оба класса не содержат изображений")

    report["class_distribution"] = {
        name: {split: int(split_counts.get(split, Counter()).get(name, 0)) for split in SPLITS}
        for name in CLASSES
    }

    if report["cross_split_exact_duplicates"]:
        report["errors"].append("Найдены точные копии файлов в разных разделах")

    metadata_path = Path(args.metadata).resolve() if args.metadata else root / "metadata.csv"
    if not metadata_path.exists():
        message = "metadata.csv не найден: контроль object_id и происхождения снимков не выполнен"
        (report["errors"] if args.require_metadata else report["warnings"]).append(message)
    else:
        object_splits: dict[str, set[str]] = defaultdict(set)
        object_classes: dict[str, set[str]] = defaultdict(set)
        rows_by_path: dict[str, dict[str, str]] = {}
        try:
            with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = set(reader.fieldnames or [])
                if not {"path", "object_id"}.issubset(fields):
                    report["errors"].append("metadata.csv должен содержать колонки path и object_id")
                else:
                    for line_number, row in enumerate(reader, start=2):
                        raw_path = (row.get("path") or "").strip().replace("\\", "/")
                        object_id = (row.get("object_id") or "").strip()
                        if not raw_path or not object_id:
                            report["errors" if args.require_metadata else "warnings"].append(
                                f"metadata.csv: пустой path/object_id в строке {line_number}"
                            )
                            continue
                        rel = Path(raw_path)
                        if rel.is_absolute() or ".." in rel.parts:
                            report["errors"].append(f"metadata.csv: небезопасный путь в строке {line_number}: {raw_path}")
                            continue
                        normalized = rel.as_posix()
                        if normalized in rows_by_path:
                            report["metadata"]["duplicate_paths"].append(normalized)
                            continue
                        rows_by_path[normalized] = row
                        parts = rel.parts
                        if len(parts) < 3 or parts[0] not in SPLITS or parts[1] not in CLASSES:
                            report["errors"].append(
                                f"metadata.csv: путь должен иметь вид split/class/file: {normalized}"
                            )
                            continue
                        object_splits[object_id].add(parts[0])
                        object_classes[object_id].add(parts[1])
                        source = (row.get("source") or "").strip()
                        license_name = (row.get("license") or "").strip()
                        if not source or not license_name:
                            report["metadata"]["missing_provenance"].append(normalized)
        except OSError as exc:
            report["errors"].append(f"Не удалось прочитать metadata.csv: {exc}")

        report["metadata"]["rows"] = len(rows_by_path)
        report["object_group_leakage"] = [
            {"object_id": object_id, "splits": sorted(splits)}
            for object_id, splits in object_splits.items() if len(splits) > 1
        ]
        mislabeled_objects = [
            {"object_id": object_id, "classes": sorted(classes)}
            for object_id, classes in object_classes.items() if len(classes) > 1
        ]
        if report["object_group_leakage"]:
            report["errors"].append("Один object_id встречается более чем в одном разделе")
        if mislabeled_objects:
            report["errors"].append(f"Один object_id размечен разными классами: {mislabeled_objects[:10]}")
        report["metadata"]["unlisted_images"] = sorted(set(all_images) - set(rows_by_path))
        report["metadata"]["missing_images"] = sorted(set(rows_by_path) - set(all_images))
        if report["metadata"]["duplicate_paths"]:
            report["errors"].append("В metadata.csv повторяются пути к изображениям")
        if report["metadata"]["unlisted_images"]:
            message = f"В metadata.csv отсутствуют записи для {len(report['metadata']['unlisted_images'])} изображений"
            (report["errors"] if args.require_metadata else report["warnings"]).append(message)
        if report["metadata"]["missing_images"]:
            report["errors"].append("metadata.csv содержит пути к изображениям, которых нет на диске")
        if report["metadata"]["missing_provenance"]:
            message = f"Не заполнены source/license у {len(report['metadata']['missing_provenance'])} изображений"
            (report["errors"] if args.require_provenance else report["warnings"]).append(message)
        if args.require_metadata and len(rows_by_path) == 0:
            report["errors"].append("metadata.csv не содержит записей")

    for split in SPLITS:
        total = sum(split_counts.get(split, Counter()).values())
        if 0 < total < 20:
            report["warnings"].append(
                f"В {split} всего {total} изображений: оценка качества может быть статистически нестабильной"
            )
    for class_name in CLASSES:
        group_count = len({
            row.get("object_id", "").strip()
            for row in rows_by_path.values()
            if row.get("object_id", "").strip()
            and Path(str(row.get("path", ""))).parts[1:2] == (class_name,)
        }) if metadata_path.exists() and "rows_by_path" in locals() else 0
        if 0 < group_count < 7:
            report["warnings"].append(
                f"Для класса {class_name} найдено только {group_count} object_id; групповое разбиение 70/15/15 может быть ненадёжным"
            )

    report["passed"] = len(report["errors"]) == 0
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== АУДИТ ДАТАСЕТА ASTROVISION AI ===")
    for split, info in report["splits"].items():
        print(f"{split}: {info['counts']} (всего {info['total']})")
    print(f"Точные дубликаты между разделами: {len(report['cross_split_exact_duplicates'])}")
    print(f"Пересечения object_id: {len(report['object_group_leakage'])}")
    print(f"Ошибки: {len(report['errors'])}; предупреждения: {len(report['warnings'])}")
    for item in report["errors"]:
        print("ОШИБКА:", item)
    for item in report["warnings"]:
        print("ПРЕДУПРЕЖДЕНИЕ:", item)
    print(f"Отчёт сохранён: {output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
