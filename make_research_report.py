"""Создаёт Markdown-отчёт только из фактических артефактов эксперимента."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def load_json(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Не найден файл: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(value, digits=4):
    return "—" if value is None else str(value) if isinstance(value, int) and not isinstance(value, bool) else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description="Сборка отчёта AstroVision AI из фактических результатов")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--repeated", default=None, help="summary.json из run_experiments.py")
    parser.add_argument("--out", default="research/RESULTS_REPORT.md")
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    experiment = load_json(args.experiment)
    audit = load_json(args.audit)
    repeated = load_json(args.repeated)

    lines = [
        "# AstroVision AI — отчёт о вычислительном эксперименте", "",
        "> Отчёт сформирован из файлов результатов. Значения не являются заявлением о научной точности за пределами указанной тестовой выборки.", "",
        "## 1. Конфигурация", "",
        f"- Дата UTC: {experiment.get('created_utc', '—')}",
        f"- Архитектура: {experiment.get('architecture', 'EfficientNet-B0')}",
        f"- Предобучение ImageNet: {experiment.get('pretrained_imagenet', '—')}",
        f"- Устройство: {experiment.get('device', '—')}",
        f"- GPU: {experiment.get('gpu_name') or 'не использовался'}",
        f"- Python: {experiment.get('python_version', '—')}; PyTorch: {experiment.get('torch_version', '—')}; torchvision: {experiment.get('torchvision_version', '—')}",
        f"- Seed: {experiment.get('seed', '—')}",
        f"- Изображения: train={experiment.get('dataset_counts', {}).get('train', '—')}, val={experiment.get('dataset_counts', {}).get('val', '—')}, test={experiment.get('dataset_counts', {}).get('test', '—')}",
        "", "## 2. Результаты на независимой тестовой выборке", "",
        "| Метрика | Значение |", "|---|---:|",
    ]
    for key, label in [
        ("n_test", "Размер test"), ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"), ("precision_macro", "Precision macro"),
        ("recall_macro", "Recall macro"), ("f1_macro", "F1 macro"),
        ("roc_auc", "ROC-AUC"), ("matthews_corrcoef", "MCC"),
        ("cohen_kappa", "Cohen's kappa"), ("brier_score", "Brier score"),
        ("ece_15_bins", "ECE (15 bins)"),
    ]:
        lines.append(f"| {label} | {fmt(metrics.get(key))} |")

    lines += ["", "### 95% bootstrap-интервалы", "", "| Метрика | Нижняя граница | Верхняя граница |", "|---|---:|---:|"]
    ci_info = metrics.get("bootstrap_95ci", {})
    for key, label in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("f1_macro", "Macro-F1")]:
        ci = ci_info.get(key)
        lines.append(f"| {label} | {fmt(ci[0]) if ci else '—'} | {fmt(ci[1]) if ci else '—'} |")
    lines += ["", f"Единица bootstrap-выборки: `{ci_info.get('bootstrap_unit', 'unknown')}`; число повторов: {ci_info.get('n_bootstrap', '—')}."]

    lines += ["", "### Метрики по классам", "", "| Класс | Precision | Recall | F1 | Число снимков |", "|---|---:|---:|---:|---:|"]
    for name, row in metrics.get("per_class", {}).items():
        lines.append(f"| {name} | {fmt(row.get('precision'))} | {fmt(row.get('recall'))} | {fmt(row.get('f1'))} | {row.get('support', '—')} |")

    cm = metrics.get("confusion_matrix")
    if cm:
        lines += ["", "### Матрица ошибок", "", "Порядок классов: " + ", ".join(experiment.get("classes", ["galaxies", "nebulae"])), "", "```text", "\n".join(" ".join(map(str, row)) for row in cm), "```"]

    if audit:
        lines += ["", "## 3. Аудит датасета", "", f"- Проверка пройдена: {audit.get('passed', False)}", f"- Ошибок: {len(audit.get('errors', []))}; предупреждений: {len(audit.get('warnings', []))}",
            f"- Точных дубликатов между разделами: {len(audit.get('cross_split_exact_duplicates', []))}",
            f"- Пересечений object_id: {len(audit.get('object_group_leakage', []))}"]
        for split, data in audit.get("splits", {}).items():
            lines.append(f"- {split}: {data.get('counts', {})} (всего {data.get('total', 0)})")
        if audit.get("errors"):
            lines += ["", "Ошибки аудита:"] + [f"- {item}" for item in audit["errors"]]
        if audit.get("warnings"):
            lines += ["", "Предупреждения аудита:"] + [f"- {item}" for item in audit["warnings"]]

    if repeated:
        lines += ["", "## 4. Повторные запуски", "", "| Метрика | Среднее | Стандартное отклонение | Число запусков |", "|---|---:|---:|---:|"]
        for key, data in repeated.get("mean_and_std", {}).items():
            lines.append(f"| {key} | {fmt(data.get('mean'))} | {fmt(data.get('std'))} | {data.get('runs', '—')} |")

    lines += ["", "## 5. Ограничения интерпретации", "",
        "1. Выводы относятся только к двум классам и распределению данных, представленным в датасете.",
        "2. Независимость тестовой выборки должна подтверждаться разделением по астрономическим объектам, а не только по файлам.",
        "3. Softmax-score не является автоматически откалиброванной вероятностью истинности.",
        "4. Классификатор не подтверждает идентичность объекта и не заменяет астрономическую экспертизу.",
        "5. При небольшом тестовом наборе доверительные интервалы могут быть широкими; требуется расширение выборки.",
        "6. Если metadata.csv содержит object_id для всех тестовых снимков, интервалы рассчитываются групповым bootstrap по объектам; иначе используется bootstrap по изображениям.", "",
        "## 6. Вывод", "",
        "Вывод о качестве модели формулируется после анализа приведённых метрик, матрицы ошибок, размера выборки и результатов повторных запусков. Не следует заявлять о превосходстве модели без заранее определённого базового метода сравнения.", ""]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт сохранён: {output.resolve()}")


if __name__ == "__main__":
    main()
