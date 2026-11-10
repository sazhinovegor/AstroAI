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

CLASSES = ("galaxies", "nebulae", "other")
SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def choose_grouped_split(
    labels: list[str], groups: list[str], seed: int = 42, attempts: int = 20
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Подбирает наиболее сбалансированное групповое разбиение 70/15/15.

    Перебирает допустимые фолды StratifiedGroupKFold с разными seed и выбирает
    вариант с минимальным отклонением долей изображений и каждого класса.
    Один object_id никогда не разделяется между выборками.
    """
    if len(labels) != len(groups) or not labels:
        raise ValueError("labels и groups должны иметь одинаковую ненулевую длину")
    if set(labels) != set(CLASSES):
        raise ValueError(f"В данных должны присутствовать классы {CLASSES}")

    group_classes: dict[str, set[str]] = defaultdict(set)
    for label, group in zip(labels, groups):
        group_classes[str(group)].add(label)
    mixed = {group: sorted(values) for group, values in group_classes.items() if len(values) != 1}
    if mixed:
        raise ValueError(f"object_id встречаются в нескольких классах: {list(mixed.items())[:10]}")

    groups_per_class = {
        name: len({g for label, g in zip(labels, groups) if label == name})
        for name in CLASSES
    }
    if any(count < 7 for count in groups_per_class.values()):
        raise ValueError(
            "Для группового разбиения с 7 фолдами требуется не менее 7 разных object_id на класс; "
            f"сейчас: {groups_per_class}"
        )

    target = {"train": 0.70, "val": 0.15, "test": 0.15}
    label_array = list(labels)
    group_array = list(groups)
    all_indices = list(range(len(labels)))
    total = len(labels)
    class_totals = {name: sum(label == name for label in labels) for name in CLASSES}

    def score_split(assigned: dict[str, list[int]]) -> float:
        score = 0.0
        for split, indices in assigned.items():
            ratio = len(indices) / total
            score += 2.0 * (ratio - target[split]) ** 2
            present = {label_array[i] for i in indices}
            if present != set(CLASSES):
                return float("inf")
            for class_name in CLASSES:
                class_ratio = sum(label_array[i] == class_name for i in indices) / class_totals[class_name]
                score += 3.0 * (class_ratio - target[split]) ** 2
        return score

    best: dict[str, list[int]] | None = None
    best_score = float("inf")
    best_seed = None

    for attempt in range(max(1, attempts)):
        outer = StratifiedGroupKFold(
            n_splits=7, shuffle=True, random_state=int(seed + attempt)
        )
        try:
            outer_folds = list(outer.split(all_indices, label_array, group_array))
        except ValueError:
            continue
        for outer_no, (train_val_idx, test_idx) in enumerate(outer_folds):
            train_val_labels = [label_array[i] for i in train_val_idx]
            train_val_groups = [group_array[i] for i in train_val_idx]
            inner = StratifiedGroupKFold(
                n_splits=6, shuffle=True,
                random_state=int(seed + 1000 + attempt * 7 + outer_no),
            )
            try:
                inner_folds = list(inner.split(
                    list(range(len(train_val_idx))), train_val_labels, train_val_groups
                ))
            except ValueError:
                continue
            for train_rel_idx, val_rel_idx in inner_folds:
                train_idx = [train_val_idx[i] for i in train_rel_idx]
                val_idx = [train_val_idx[i] for i in val_rel_idx]
                assigned = {"train": train_idx, "val": val_idx, "test": list(test_idx)}
                current_score = score_split(assigned)
                if current_score < best_score:
                    best, best_score = assigned, current_score
                    best_seed = {
                        "outer_random_state": int(seed + attempt),
                        "inner_random_state": int(seed + 1000 + attempt * 7 + outer_no),
                        "outer_fold": outer_no,
                    }

    if best is None:
        raise ValueError(
            "Не удалось построить разбиение, где оба класса присутствуют в train/val/test. "
            "Увеличьте число независимых object_id для каждого класса."
        )

    # Финальная инвариантная проверка: object_id не пересекаются.
    object_sets = {
        split: {group_array[i] for i in indices}
        for split, indices in best.items()
    }
    if (object_sets["train"] & object_sets["val"] or
            object_sets["train"] & object_sets["test"] or
            object_sets["val"] & object_sets["test"]):
        raise RuntimeError("Внутренняя ошибка: обнаружено пересечение object_id")

    details = {
        "seed": int(seed),
        "selection_score": float(best_score),
        "selected_candidate": best_seed,
        "target_ratios": target,
        "image_ratios": {split: len(indices) / total for split, indices in best.items()},
        "class_image_ratios": {
            split: {
                class_name: sum(label_array[i] == class_name for i in indices) / class_totals[class_name]
                for class_name in CLASSES
            }
            for split, indices in best.items()
        },
    }
    return best, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Групповое разбиение датасета AstroVision AI")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--raw_dir", default="raw")
    parser.add_argument("--metadata", default="raw_metadata.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=20,
                        help="Число random seed для поиска сбалансированного группового разбиения")
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

    try:
        assigned, split_details = choose_grouped_split(
            [r["class"] for r in records], [r["object_id"] for r in records],
            seed=args.seed, attempts=args.attempts,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    existing = [root / split for split in SPLITS if (root / split).exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Папки разбиения уже существуют: {existing}. Добавьте --overwrite, если их можно заменить."
        )
    if args.overwrite:
        for folder in existing:
            shutil.rmtree(folder)

    output_metadata = []
    summary: dict[str, object] = {
        **split_details, "split_counts": {}, "object_counts": {}, "ratios": {}
    }
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
    print("=== ГОТОВО: СБАЛАНСИРОВАННОЕ РАЗБИЕНИЕ ПО АСТРОНОМИЧЕСКИМ ОБЪЕКТАМ ===")
    for split in SPLITS:
        print(
            f"{split}: {summary['split_counts'][split]}, доля={summary['ratios'][split]:.1%}, "
            f"object_id={summary['object_counts'][split]}"
        )
    print(f"Критерий баланса: {summary['selection_score']:.6f}")
    print(f"metadata.csv: {out_meta}")
    print(f"Отчёт разбиения: {report_path}")
    if missing_provenance:
        print(f"ПРЕДУПРЕЖДЕНИЕ: у {missing_provenance} снимков не заполнены source/license")
    print("Следующий шаг: python audit_dataset.py --data_dir dataset --require_metadata --require_provenance")


if __name__ == "__main__":
    main()
