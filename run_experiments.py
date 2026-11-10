"""Повторяет обучение AstroVision AI с несколькими seed и сводит результаты.

Пример:
python run_experiments.py --data_dir dataset --epochs 15 --batch_size 16 --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

METRICS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
           "f1_macro", "roc_auc", "matthews_corrcoef", "ece_15_bins"]


def main():
    parser = argparse.ArgumentParser(description="Повторные эксперименты AstroVision AI")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--architectures", nargs="+", choices=["efficientnet_b0", "resnet50", "convnext_tiny"],
                        default=["efficientnet_b0", "resnet50", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--out_dir", default="research_runs")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--class_weighted_loss", action="store_true")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Сравнить выбранные архитектуры без предобученных весов")
    parser.add_argument("--require_metadata", action="store_true")
    parser.add_argument("--require_provenance", action="store_true")
    args = parser.parse_args()
    if args.from_scratch and args.freeze_epochs > 0:
        parser.error("для --from_scratch задайте --freeze_epochs 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for arch in args.architectures:
      for seed in args.seeds:
        run_dir = out_dir / arch / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "astrovision_model.pth"
        command = [
            sys.executable, "train.py", "--data_dir", args.data_dir,
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--lr", str(args.lr), "--img_size", str(args.img_size),
            "--freeze_epochs", str(args.freeze_epochs), "--patience", str(args.patience),
            "--seed", str(seed), "--arch", arch, "--out", str(model_path),
        ]
        if args.no_amp:
            command.append("--no_amp")
        if args.no_aug:
            command.append("--no_aug")
        if args.class_weighted_loss:
            command.append("--class_weighted_loss")
        if args.from_scratch:
            command.append("--from_scratch")
        if args.require_metadata:
            command.append("--require_metadata")
        if args.require_provenance:
            command.append("--require_provenance")
        log_path = run_dir / "training.log"
        print(f"\n=== Запуск seed={seed} ===")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=Path(__file__).resolve().parent,
                                    stdout=log, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            print(f"Ошибка запуска seed={seed}. См. {log_path}")
            raise SystemExit(result.returncode)
        metrics_path = model_path.with_suffix("").with_suffix(".metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"architecture": arch, "seed": seed, **{key: metrics.get(key) for key in METRICS}}
        records.append(row)
        print(f"arch={arch}, seed={seed}: accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

    summary_path = out_dir / "repeated_experiments.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["architecture", "seed", *METRICS])
        writer.writeheader()
        writer.writerows(records)

    aggregate = {}
    for arch in args.architectures:
        aggregate[arch] = {}
        arch_records = [r for r in records if r["architecture"] == arch]
        for metric in METRICS:
            values = [float(r[metric]) for r in arch_records if r.get(metric) is not None]
            if values:
                mean = sum(values) / len(values)
                aggregate[arch][metric] = {
                    "mean": mean,
                    "std": (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5
                            if len(values) > 1 else 0.0,
                    "runs": len(values),
                }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСводка сохранена: {summary_path} и {out_dir / 'summary.json'}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
