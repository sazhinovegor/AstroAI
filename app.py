from pathlib import Path
import json, subprocess, sys, time
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "astrovision_model.pth"
PROGRESS_PATH = ROOT / "training_progress.json"
LOG_PATH = ROOT / "training_log.txt"
STOP_FLAG = ROOT / "stop_training.flag"
DISPLAY = {"galaxies": "Галактика", "nebulae": "Туманность"}

st.set_page_config(page_title="AstroVision AI", page_icon="🌌", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at top, #17233b 0%, #0b1020 48%, #080b14 100%); color:#edf2ff;}
section[data-testid="stSidebar"] {background:#10182a;}
.hero {padding: 1.6rem 1.8rem; border:1px solid #293b5c; border-radius:18px;
       background:linear-gradient(120deg,rgba(36,68,112,.55),rgba(20,28,49,.7));}
.muted {color:#aab8d1;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path_str, mtime):
    checkpoint = torch.load(path_str, map_location="cpu", weights_only=False)
    img_size = checkpoint.get("img_size", 224)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["classes"], checkpoint.get("threshold", 0.65),         checkpoint.get("metrics", {}), checkpoint.get("history", []), img_size


@st.cache_resource
def benchmark_inference(path_str, mtime):
    """Скорость инференса на CPU: миллисекунды на одно изображение."""
    model, _, _, _, _, img_size = load_model(path_str, mtime)
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        n, t0 = 20, time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t0) / n * 1000


def image_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def read_progress():
    if not PROGRESS_PATH.exists():
        return {}
    try:
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def log_tail(n=25):
    if not LOG_PATH.exists():
        return ""
    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def split_counts(split):
    d = ROOT / "dataset" / split
    out = {}
    if d.exists():
        for c in sorted(p.name for p in d.iterdir() if p.is_dir()):
            out[c] = sum(1 for e in ("*.jpg", "*.jpeg", "*.png") for _ in (d / c).glob(e))
    return out


st.markdown('<div class="hero"><h1>🌌 AstroVision AI</h1><p class="muted">Интеллектуальный анализ астрофотографий · Галактики и туманности · v0.3</p></div>', unsafe_allow_html=True)
st.write("")
page = st.sidebar.radio("Навигация", ["Анализ снимка", "Обучение модели", "Научная статистика", "О проекте"])
st.sidebar.caption("Исследовательский прототип • версия 0.2")

# =====================================================================
if page == "Анализ снимка":
    st.subheader("Анализ астрофотографии")
    st.write("Загрузите изображение в формате JPG, JPEG или PNG. Модель выдаст класс и оценку уверенности.")
    tta = st.checkbox("TTA: усреднение с отражённым снимком (точнее, чуть медленнее)", value=False)
    uploaded = st.file_uploader("Выберите астрофотографию", type=["jpg", "jpeg", "png"])
    if not MODEL_PATH.exists():
        st.info("Модель ещё не обучена. Загрузите датасет и запустите обучение на странице «Обучение модели» или выполните:")
        st.code("python train.py --data_dir dataset --epochs 8")
    elif uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Загруженное изображение", use_container_width=True)
        try:
            model, classes, threshold, metrics, history, img_size = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            x = image_transform(img_size)(image).unsqueeze(0)
            t0 = time.perf_counter()
            with torch.no_grad():
                if tta:
                    x2 = torch.flip(x, dims=[3])
                    probs = torch.softmax((model(x) + model(x2)) / 2, dim=1)[0]
                else:
                    probs = torch.softmax(model(x), dim=1)[0]
            infer_ms = (time.perf_counter() - t0) * 1000
            values = probs.tolist()
            idx = int(torch.argmax(probs))
            confidence = values[idx]
            with right:
                if confidence < threshold:
                    st.warning("Неизвестный объект / низкая уверенность")
                else:
                    st.success(f"Предсказание: **{DISPLAY.get(classes[idx], classes[idx])}**")
                st.metric("Уверенность модели", f"{confidence*100:.1f}%")
                st.caption(f"⏱ Время инференса: {infer_ms:.0f} мс (CPU)")
                chart = pd.DataFrame({"Класс": [DISPLAY.get(c, c) for c in classes], "Вероятность": [v*100 for v in values]})
                fig = px.bar(chart, x="Класс", y="Вероятность", range_y=[0, 100], text_auto=".1f",
                             color="Класс", title="Распределение вероятностей")
                fig.update_layout(template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Вероятность — оценка модели, а не подтверждённая вероятность истинности. Результат зависит от состава обучающих данных.")
        except Exception as e:
            st.error(f"Не удалось загрузить модель или обработать изображение: {e}")

# =====================================================================
elif page == "Обучение модели":
    st.subheader("Обучение модели")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = st.session_state.get("train_proc")

    if proc is not None and proc.poll() is None:
        # --- обучение идёт: показываем живой прогресс и обновляемся rerun'ами ---
        payload = read_progress()
        ep, eps = payload.get("epoch", 0), max(payload.get("epochs", 1), 1)
        st.progress(min(ep / eps, 1.0), text=f"Эпоха {ep}/{eps}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train loss", f"{payload.get('train_loss', float('nan')):.4f}")
        c2.metric("Val loss", f"{payload.get('val_loss', float('nan')):.4f}")
        c3.metric("Val accuracy", f"{payload.get('val_acc', 0)*100:.1f}%")
        c4.metric("Скорость", f"{payload.get('img_per_sec', 0):.1f} изобр/с")
        hist = payload.get("history", [])
        if hist:
            fig = px.line(pd.DataFrame(hist).set_index("epoch")[["train_loss", "val_loss"]],
                          labels={"value": "Loss", "epoch": "Эпоха"}, title="Динамика функции потерь")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        st.code(log_tail(), language=None)
        if st.button("🛑 Остановить после текущей эпохи", type="secondary"):
            STOP_FLAG.write_text("stop", encoding="utf-8")
            st.toast("Сигнал остановки отправлен — модель сохранится после завершения эпохи.")
        time.sleep(1.5)
        st.rerun()

    if proc is not None and proc.poll() is not None:
        # --- обучение завершилось ---
        rc = proc.returncode
        payload = read_progress()
        st.session_state["train_proc"] = None
        if rc == 0:
            st.success("Обучение завершено ✅")
            m = payload.get("metrics") or {}
            if m:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Test accuracy", f"{m.get('accuracy', 0):.3f}")
                c2.metric("Precision (macro)", f"{m.get('precision_macro', 0):.3f}")
                c3.metric("Recall (macro)", f"{m.get('recall_macro', 0):.3f}")
                c4.metric("F1-score (macro)", f"{m.get('f1_macro', 0):.3f}")
            st.info("Обновлённая модель уже подключена к странице «Анализ снимка».")
        else:
            st.error(f"Обучение завершилось с ошибкой (код {rc}). Последние строки лога:")
            st.code(log_tail(40), language=None)
        st.code(log_tail(), language=None)

    if st.session_state.get("train_proc") is None:
        # --- форма запуска ---
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Состав датасета** (`dataset/`)")
            for split in ("train", "val", "test"):
                counts = split_counts(split)
                if counts:
                    st.write(f"`{split}`: " + ", ".join(f"{k} — {v}" for k, v in counts.items()))
                else:
                    st.write(f"`{split}`: — папка не заполнена")
            ok = bool(split_counts("train")) and bool(split_counts("val"))
            if not ok:
                st.warning("Нужны непустые папки dataset/train и dataset/val. См. README.md.")
        with c2:
            st.markdown("**Вычислительное устройство**")
            if device.type == "cuda":
                st.success(f"🚀 GPU: {torch.cuda.get_device_name(0)} — обучение с AMP (float16)")
            else:
                st.info("💻 CPU: обучение возможно, но медленное (AMP bfloat16 включён). Рекомендуется GPU.")
            st.caption("Скорость обучения и инференса зависит от устройства, размера модели и batch size.")

        st.markdown("**Гиперпараметры**")
        c1, c2, c3 = st.columns(3)
        epochs = c1.slider("Эпохи", 2, 50, 8)
        batch_size = c2.select_slider("Batch size", options=[4, 8, 16, 32], value=16)
        lr = c3.select_slider("Learning rate", options=[1e-5, 3e-5, 1e-4, 3e-4], value=1e-4, format_func=lambda v: f"{v:.0e}")
        c1, c2, c3 = st.columns(3)
        freeze_epochs = c1.slider("Эпох с замороженным backbone", 0, 5, 2)
        patience = c2.slider("Ранняя остановка (эпох без улучшения)", 0, 10, 4)
        threshold = c3.slider("Порог «неизвестный объект»", 0.3, 0.95, 0.65, 0.05)
        with st.expander("Дополнительно"):
            c1, c2, c3 = st.columns(3)
            use_amp = c1.checkbox("AMP (ускорение)", value=True)
            use_aug = c2.checkbox("Расширенные аугментации", value=True)
            workers = c3.number_input("Workers (0 — безопаснее на Windows)", 0, 8, 0)
            img_size = c1.select_slider("Размер изображения", options=[160, 224, 256], value=224)

        if st.button("🚀 Запустить обучение", type="primary", disabled=not ok):
            cmd = [sys.executable, str(ROOT / "train.py"), "--data_dir", str(ROOT / "dataset"),
                   "--epochs", str(epochs), "--batch_size", str(batch_size), "--lr", str(lr),
                   "--threshold", str(threshold), "--freeze_epochs", str(freeze_epochs),
                   "--patience", str(patience), "--workers", str(workers),
                   "--img_size", str(img_size), "--progress_file", str(PROGRESS_PATH),
                   "--stop_file", str(STOP_FLAG), "--out", str(MODEL_PATH)]
            if not use_amp:
                cmd.append("--no_amp")
            if not use_aug:
                cmd.append("--no_aug")
            PROGRESS_PATH.unlink(missing_ok=True)
            LOG_PATH.write_text("", encoding="utf-8")
            STOP_FLAG.unlink(missing_ok=True)
            logf = open(LOG_PATH, "a", encoding="utf-8")
            new_proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
            logf.close()
            st.session_state["train_proc"] = new_proc
            st.toast("Обучение запущено!")
            time.sleep(1)
            st.rerun()

# =====================================================================
elif page == "Научная статистика":
    st.subheader("Научная статистика")
    if not MODEL_PATH.exists():
        st.info("После обучения модели здесь появятся метрики независимого тестирования.")
    else:
        model, classes, threshold, metrics, history, img_size = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        ms = benchmark_inference(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        c0, c1, c2, c3 = st.columns(4)
        c0.metric("Скорость инференса", f"{ms:.0f} мс")
        c1.metric("Производительность", f"{1000/ms:.1f} изобр/с")
        c2.metric("Размер входа", f"{img_size}×{img_size}")
        c3.metric("Устройство теста", "CPU")
        st.caption("Бенчмарк: одиночные изображения на CPU. На GPU инференс обычно в разы быстрее.")
        st.caption("Метрики рассчитаны скриптом обучения на папке test, если она была предоставлена.")
        if history:
            fig = px.line(pd.DataFrame(history).set_index("epoch")[["train_loss", "val_loss", "train_acc", "val_acc"]],
                          labels={"value": "Значение", "epoch": "Эпоха"}, title="История обучения")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        if metrics:
            cols = st.columns(4)
            for col, key, label in zip(cols, ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                                       ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]):
                col.metric(label, f"{metrics.get(key, 0):.3f}")
            cm = metrics.get("confusion_matrix")
            if cm:
                import plotly.figure_factory as ff
                fig = ff.create_annotated_heatmap(z=cm, x=[DISPLAY.get(c, c) for c in classes],
                    y=[DISPLAY.get(c, c) for c in classes], colorscale="Blues", showscale=True)
                fig.update_layout(title="Матрица ошибок", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            pc = metrics.get("per_class")
            if pc:
                st.markdown("**Метрики по классам**")
                st.dataframe(pd.DataFrame(pc).T, use_container_width=True)
        else:
            st.warning("Тестовые метрики отсутствуют. Проверьте структуру test/ и повторите обучение.")
        st.caption(f"Порог для ответа «неизвестный объект»: {threshold:.2f}")

# =====================================================================
else:
    st.subheader("О проекте")
    st.markdown("""
**AstroVision AI v0.3** — учебно-исследовательский прототип компьютерного зрения для классификации астрофотографий на два класса: галактики и туманности.

**Метод:** transfer learning на EfficientNet-B0 (ImageNet) с двухэтапным дообучением: сначала замороженный backbone и обучаемый классификатор, затем разморозка с малым learning rate.

**Скорость (v0.3):**
- Mixed precision (AMP): float16 на GPU, bfloat16 на CPU;
- `channels_last`, `pin_memory`, `non_blocking`, `cudnn.benchmark`;
- косинусное расписание learning rate с прогревом;
- замер скорости обучения (изображений/с) и бенчмарк инференса;
- опциональный TTA (test-time augmentation) — усреднение предсказаний с отражённым снимком.

**Обучение (v0.3):**
- запуск обучения прямо из веб-интерфейса с живым прогрессом и графиком потерь;
- мягкая остановка после текущей эпохи; ранняя остановка по val_loss;
- расширенные аугментации (повороты, отражения, кропы, цветовые сдвиги, размытие);
- per-class метрики и история обучения в чекпоинте.

**Ограничения:** модель не определяет название конкретного небесного объекта, координаты или физические свойства. Она классифицирует изображение по двум заданным классам. Значение softmax — оценка модели, не автоматически откалиброванная вероятность истинности; порог «неизвестный объект» подбирается на validation set.

**Технологии:** Python, PyTorch, torchvision, Streamlit, Plotly, scikit-learn.

См. README.md: установка, структура датасета и публикация на Streamlit Community Cloud.
""")
