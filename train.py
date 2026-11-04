"""AstroVision AI — обучение классификатора астрофотографий (версия 0.3).

Что нового по сравнению со стартовой версией:
- Скорость: mixed precision (AMP), channels_last, pin_memory/non_blocking,
  cudnn.benchmark, cosine-расписание с прогревом, замер изображений/сек.
- Обучение: корректное замораживание/размораживание backbone с отдельным
  расписанием learning rate, ранняя остановка, аугментации и per-class метрики.
- Веб-интерфейс: прогресс пишется в JSON (--progress_file), чтобы Streamlit
  мог показывать живой прогресс; мягкая остановка через --stop_file.
"""
from pathlib import Path
import argparse, copy, json, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transforms(img_size=224, augment=True):
    """Трансформации. Астроснимки не имеют «верха», поэтому вертикальное
    отражение и повороты — законные аугментации."""
    train_ops = [transforms.Resize((img_size, img_size))]
    if augment:
        train_ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
        ]
    train_ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    eval_ops = [transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def write_progress(path, payload):
    """Атомарно пишем JSON-прогресс для веб-интерфейса."""
    if not path:
        return
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              amp_dtype=None, non_blocking=False, channels_last=False):
    """Одна эпоха (train или eval). Возвращает (loss, accuracy, секунд)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
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
    return total_loss / max(total, 1), correct / max(total, 1), time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description="Train AstroVision AI classifier (v0.2)")
    p.add_argument("--data_dir", default="dataset")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--workers", type=int, default=0, help="0 — безопаснее всего на Windows")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_epochs", type=int, default=2,
                   help="сколько первых эпох обучать только классификатор")
    p.add_argument("--patience", type=int, default=4,
                   help="ранняя остановка: эпох без улучшения val_loss (0 = выкл)")
    p.add_argument("--no_amp", action="store_true", help="отключить mixed precision")
    p.add_argument("--no_aug", action="store_true", help="отключить расширенные аугментации")
    p.add_argument("--progress_file", default=None, help="JSON-файл живого прогресса")
    p.add_argument("--stop_file", default=None, help="флаг мягкой остановки")
    p.add_argument("--out", default="models/astrovision_model.pth")
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True  # фиксированный размер входа -> ускорение на GPU

    root = Path(args.data_dir)
    train_dir, val_dir, test_dir = root / "train", root / "val", root / "test"
    for d in (train_dir, val_dir):
        if not d.exists():
            raise SystemExit(f"Не найдена папка {d}. См. README.md.")
    train_tf, eval_tf = build_transforms(args.img_size, augment=not args.no_aug)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    if train_ds.classes != val_ds.classes:
        raise SystemExit(f"Классы train {train_ds.classes} не совпадают с val {val_ds.classes}")
    if len(train_ds.classes) != 2:
        raise SystemExit("Для первой версии нужны ровно два класса: galaxies и nebulae.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    use_amp = (not args.no_amp) and device.type in ("cuda", "cpu")
    amp_dtype = (torch.float16 if device.type == "cuda" else
                 torch.bfloat16 if device.type == "cpu" else None) if use_amp else None
    channels_last = device.type == "cuda"
    print(f"Устройство: {device} | AMP: {use_amp} | Классы: {train_ds.classes} | "
          f"train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin,
                            persistent_workers=args.workers > 0)

    # Transfer learning: предобученный backbone + новый классификатор.
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_ds.classes))
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()

    # Две группы параметров: классификатор учится быстрее,
    # backbone до размораживания полностью заморожен.
    head_params = list(model.classifier.parameters())
    head_ids = {id(pr) for pr in head_params}
    backbone_params = [pr for pr in model.parameters() if id(pr) not in head_ids]

    if args.freeze_epochs > 0:
        for pr in backbone_params:
            pr.requires_grad_(False)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    # Общий график: прогрев, затем косинусное снижение.
    # Для backbone множитель равен нулю до размораживания.
    warmup = 1
    total = max(args.epochs, 1)

    def schedule_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup + 1) / max(total - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    def backbone_factor(epoch):
        return 0.0 if epoch < args.freeze_epochs else schedule_factor(epoch)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[backbone_factor, schedule_factor]
    )
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    best_loss, best_state = float("inf"), copy.deepcopy(model.state_dict())
    history, wait = [], 0
    payload = {"status": "running", "epochs": args.epochs, "epoch": 0,
               "classes": train_ds.classes, "device": str(device), "amp": use_amp}
    write_progress(args.progress_file, payload)

    for epoch in range(args.epochs):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for pr in backbone_params:
                pr.requires_grad_(True)
            print(f"Эпоха {epoch+1}: backbone разморожен; используется lr в 10 раз ниже классификатора")

        tr_loss, tr_acc, tr_sec = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_dtype,
            non_blocking=pin, channels_last=channels_last)
        va_loss, va_acc, _ = run_epoch(
            model, val_loader, criterion, device, amp_dtype=amp_dtype,
            non_blocking=pin, channels_last=channels_last)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        img_per_sec = len(train_ds) / max(tr_sec, 1e-6)
        print(f"Эпоха {epoch+1}/{args.epochs}: train_loss={tr_loss:.4f} (acc {tr_acc:.3f}), "
              f"val_loss={va_loss:.4f} (acc {va_acc:.3f}), lr={lr_now:.2e}, "
              f"{tr_sec:.1f} с, {img_per_sec:.1f} изобр/с")

        history.append({"epoch": epoch + 1, "train_loss": round(tr_loss, 5),
                        "val_loss": round(va_loss, 5), "train_acc": round(tr_acc, 4),
                        "val_acc": round(va_acc, 4), "lr": lr_now,
                        "epoch_sec": round(tr_sec, 2), "img_per_sec": round(img_per_sec, 1)})

        if va_loss < best_loss - 1e-4:
            best_loss, wait = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        payload.update({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_acc, "val_acc": va_acc, "lr": lr_now,
                        "epoch_sec": tr_sec, "img_per_sec": img_per_sec,
                        "best_val_loss": best_loss, "history": history})
        write_progress(args.progress_file, payload)

        # Мягкая остановка из веб-интерфейса (после завершения эпохи).
        if args.stop_file and Path(args.stop_file).exists():
            Path(args.stop_file).unlink()
            print("Получен сигнал остановки: сохраняем текущую лучшую модель.")
            payload.update({"status": "stopped", "message": "Остановлено пользователем"})
            write_progress(args.progress_file, payload)
            break

        if args.patience > 0 and wait >= args.patience:
            print(f"Ранняя остановка: val_loss не улучшается {args.patience} эпох.")
            payload.update({"status": "early_stopped",
                            "message": f"Early stopping после {epoch+1} эпох"})
            write_progress(args.progress_file, payload)
            break

    model.load_state_dict(best_state)
    metrics = {}
    if test_dir.exists():
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
        if len(test_ds) == 0:
            raise SystemExit("Папка test существует, но не содержит изображений.")
        if test_ds.classes != train_ds.classes:
            raise SystemExit("Классы test не совпадают с train.")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=pin)
        ys, preds = [], []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    out = model(x.to(device, non_blocking=pin))
                preds.extend(out.argmax(1).cpu().tolist()); ys.extend(y.tolist())
        labels = list(range(len(train_ds.classes)))
        precision, recall, f1, _ = precision_recall_fscore_support(
            ys, preds, average="macro", zero_division=0)
        pc_p, pc_r, pc_f, pc_s = precision_recall_fscore_support(
            ys, preds, labels=labels, zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(ys, preds)),
            "precision_macro": float(precision), "recall_macro": float(recall),
            "f1_macro": float(f1),
            "confusion_matrix": confusion_matrix(ys, preds, labels=labels).tolist(),
            "per_class": {train_ds.classes[i]: {
                "precision": float(pc_p[i]), "recall": float(pc_r[i]),
                "f1": float(pc_f[i]), "support": int(pc_s[i])} for i in labels},
        }
        print("Тестовые метрики:", json.dumps(metrics, ensure_ascii=False))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.cpu().state_dict(), "classes": train_ds.classes,
                "threshold": args.threshold, "metrics": metrics, "history": history,
                "img_size": args.img_size, "arch": "efficientnet_b0",
                "amp": use_amp, "best_val_loss": best_loss}, out)
    payload.update({"status": payload.get("status") if payload.get("status") not in
                    ("running",) else "done", "metrics": metrics,
                    "model_path": str(out)})
    if payload.get("status") == "running":
        payload["status"] = "done"
    write_progress(args.progress_file, payload)
    print(f"Модель сохранена: {out}")


if __name__ == "__main__":
    main()
