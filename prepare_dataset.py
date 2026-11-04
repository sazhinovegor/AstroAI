"""Создаёт train/val/test без разделения одного астрономического объекта.

Вход: dataset/raw/<class>/... и dataset/raw_metadata.csv.
Выход: dataset/{train,val,test}/<class>/... + dataset/metadata.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ("galaxies", "nebulae")
SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Групповое разбиение датасета AstroVision AI")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--raw_dir", default="raw")
    parser.add_argument("--metadata", default="raw_metadata.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    raw_root = (root / args.raw_dir).resolve()
    metadata_path = (root / args.metadata).resolve()
    if not metadata_path.is_file():
        raise SystemExit(f"Не найден входной CSV: {metadata_path}")

    records: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    object_classes: dict[str, set[str]] = defaultdict(set)
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "class", "object_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise SystemExit("raw_metadata.csv должен иметь колонки: path,class,object_id; желательно source,license")
        for line, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip().replace("\\", "/")
            class_name = (row.get("class") or "").strip().lower()
            object_id = (row.get("object_id") or "").strip()
            if class_name not in CLASSES or not raw_path or not object_id:
                raise SystemExit(f"Проверьте path/class/object_id в строке {line}")
            rel = Path(raw_path)
            if rel.is_absolute() or ".." in rel.parts:
                raise SystemExit(f"Недопустимый путь в строке {line}: {raw_path}")
            source_path = (root / rel).resolve()
            if not source_path.is_relative_to(raw_root / class_name) or not source_path.is_file():
                raise SystemExit(
                    f"Файл отсутствует или находится не в каталоге raw/{class_name}: {source_path}"
                )
            if source_path.suffix.lower() not in EXTENSIONS:
                raise SystemExit(f"Неподдерживаемое расширение файла: {source_path}")
            normalized = source_path.relative_to(root).as_posix()
            if normalized in seen_paths:
                raise SystemExit(f"Путь повторяется в CSV: {normalized}")
            seen_paths.add(normalized)
            object_classes[object_id].add(class_name)
            records.append({
                "source_path": str(source_path),
                "raw_path": normalized,
                "class": class_name,
                "object_id": object_id,
                "source": (row.get("source") or "").strip(),
                "license": (row.get("license") or "").strip(),
            })

    if not records:
        raise SystemExit("В raw_metadata.csv нет записей")
    multi_class_objects = {obj: sorted(classes) for obj, classes in object_classes.items() if len(classes) != 1}
    if multi_class_objects:
        raise SystemExit(f"Один object_id отнесён к разным классам: {list(multi_class_objects.items())[:10]}")

    groups_per_class = {
        class_name: len({r["object_id"] for r in records if r["class"] == class_name})
        for class_name in CLASSES
    }
    if any(count < 7 for count in groups_per_class.values()):
        raise SystemExit(
            "Для группового разбиения примерно 70/15/15 желательно не менее 7 разных object_id "
            f"на класс. Сейчас: {groups_per_class}. Не делите фотографии одного объекта между разделами."
        )

    existing = [root / split for split in SPLITS if (root / split).exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Папки разбиения уже существуют: {existing}. Добавьте --overwrite, если их можно заменить."
        )
    if args.overwrite:
        for folder in existing:
            shutil.rmtree(folder)

    labels = [r["class"] for r in records]
    groups = [r["object_id"] for r in records]
    indices = list(range(len(records)))

    # Первая групповая стратификация: один из семи фолдов (около 14%) — test.
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=args.seed)
    train_val_idx, test_idx = next(outer.split(indices, labels, groups))

    # В оставшейся части выделяем около 1/6 под validation (~14% всего набора).
    tv_labels = [labels[i] for i in train_val_idx]
    tv_groups = [groups[i] for i in train_val_idx]
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=args.seed + 1)
    train_rel_idx, val_rel_idx = next(inner.split(train_val_idx, tv_labels, tv_groups))
    train_idx = [train_val_idx[i] for i in train_rel_idx]
    val_idx = [train_val_idx[i] for i in val_rel_idx]

    assigned = {"train": train_idx, "val": val_idx, "test": list(test_idx)}
    for split, split_indices in assigned.items():
        split_classes = {labels[i] for i in split_indices}
        if split_classes != set(CLASSES):
            raise SystemExit(
                f"StratifiedGroupKFold не распределил оба класса в {split}: {split_classes}. "
                "Нужно больше разных астрономических объектов каждого класса."
            )

    output_metadata = []
    summary = {"seed": args.seed, "split_counts": {}, "object_counts": {}, "ratios": {}}
    for split, split_indices in assigned.items():
        class_counts = Counter()
        split_objects = defaultdict(set)
        for index in split_indices:
            record = records[index]
            rel_source = Path(record["source_path"]).relative_to(raw_root / record["class"])
            destination = root / split / record["class"] / rel_source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["source_path"], destination)
            rel_destination = destination.relative_to(root).as_posix()
            output_metadata.append({
                "path": rel_destination,
                "object_id": record["object_id"],
                "source": record["source"],
                "license": record["license"],
            })
            class_counts[record["class"]] += 1
            split_objects[record["class"]].add(record["object_id"])
        summary["split_counts"][split] = dict(class_counts)
        summary["object_counts"][split] = {k: len(v) for k, v in split_objects.items()}
        summary["ratios"][split] = len(split_indices) / len(records)

    out_meta = root / "metadata.csv"
    with out_meta.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "object_id", "source", "license"])
        writer.writeheader()
        writer.writerows(sorted(output_metadata, key=lambda row: row["path"]))
    report_path = root / "split_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_provenance = sum(1 for row in output_metadata if not row["source"] or not row["license"])
    print("=== ГОТОВО: РАЗБИЕНИЕ ПО АСТРОНОМИЧЕСКИМ ОБЪЕКТАМ ===")
    for split in SPLITS:
        print(f"{split}: {summary['split_counts'][split]}, доля={summary['ratios'][split]:.1%}, object_id={summary['object_counts'][split]}")
    print(f"metadata.csv: {out_meta}")
    print(f"Отчёт разбиения: {report_path}")
    if missing_provenance:
        print(f"ПРЕДУПРЕЖДЕНИЕ: у {missing_provenance} снимков не заполнены source/license")
    print("Следующий шаг: python audit_dataset.py --data_dir dataset --require_metadata --require_provenance")


if __name__ == "__main__":
    main()
