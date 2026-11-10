from pathlib import Path
import json, subprocess, sys, time
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageOps
from model_utils import build_eval_transform
from model_factory import build_model, ARCHITECTURES
from explainability import make_gradcam
from inference_utils import predict_many, predict_image
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "astrovision_model.pth"
PROGRESS_PATH = ROOT / "training_progress.json"
LOG_PATH = ROOT / "training_log.txt"
STOP_FLAG = ROOT / "stop_training.flag"
DISPLAY = {"galaxies": "Галактика", "nebulae": "Туманность", "other": "Не подходит"}

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
    checkpoint = torch.load(path_str, map_location="cpu", weights_only=True)
    img_size = checkpoint.get("img_size", 224)
    arch = checkpoint.get("arch", "efficientnet_b0")
    model = build_model(arch, len(checkpoint["classes"]), pretrained=False)
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
    # Та же предобработка, что использовалась на validation/test.
    return build_eval_transform(img_size)


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
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if d.exists():
        for class_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            out[class_dir.name] = sum(
                1 for file_path in class_dir.rglob("*")
                if file_path.is_file() and file_path.suffix.lower() in extensions
            )
    return out


st.markdown('<div class="hero"><h1>🌌 AstroVision AI</h1><p class="muted">Исследовательская система классификации астрофотографий · Только галактики и туманности · open-set фильтрация · v2.0</p></div>', unsafe_allow_html=True)
st.write("")
page = st.sidebar.radio("Навигация", ["Анализ снимка", "Пакетный анализ", "Обучение модели", "Научная статистика", "О проекте"])
st.sidebar.caption("Исследовательский прототип • версия 1.0")

# =====================================================================
if page == "Анализ снимка":
    st.subheader("Анализ астрофотографии")
    st.write("Загрузите изображение. Система принимает только галактики и туманности. Любые другие объекты должны отклоняться.")
    tta = st.checkbox("TTA: усреднение вероятностей с отражённым снимком (медленнее)", value=False)
    show_gradcam = st.checkbox("Показать Grad-CAM (визуальное объяснение модели)", value=False)
    uploaded = st.file_uploader("Выберите астрофотографию", type=["jpg", "jpeg", "png"])
    if not MODEL_PATH.exists():
        st.info("Модель ещё не обучена. Загрузите датасет и запустите обучение на странице «Обучение модели» или выполните:")
        st.code("python train.py --data_dir dataset --epochs 8")
    elif uploaded:
        if uploaded.size > 20 * 1024 * 1024:
            st.error("Файл больше 20 МБ. Уменьшите изображение и попробуйте снова.")
            st.stop()
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Загруженное изображение", use_container_width=True)
        try:
            model, classes, threshold, metrics, history, img_size = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            result = predict_image(model, image, classes, img_size, threshold, tta)
            values = [result.probabilities[c] for c in classes]
            idx = classes.index(result.predicted_class)
            confidence = result.confidence
            with right:
                if result.status == "not_supported":
                    st.error("ОШИБКА: изображение не относится к двум поддерживаемым категориям — галактикам или туманностям. Загрузите подходящую астрофотографию.")
                    st.caption(result.rejection_reason)
                else:
                    st.success(f"Объект распознан: **{DISPLAY.get(classes[idx], classes[idx])}**")
                st.metric("Оценка модели", f"{confidence*100:.1f}%")
                st.caption(f"⏱ Время анализа: {result.inference_ms:.0f} мс")
                chart = pd.DataFrame({"Класс": [DISPLAY.get(c, c) for c in classes], "Оценка": [v*100 for v in values]})
                fig = px.bar(chart, x="Класс", y="Оценка", range_y=[0, 100], text_auto=".1f",
                             color="Класс", title="Распределение оценок модели")
                fig.update_layout(template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Оценки softmax не являются вероятностями истинности. Отказ зависит от качества и разнообразия обучающего класса other.")
            if show_gradcam:
                with st.spinner("Строю карту Grad-CAM..."):
                    overlay, cam_class, cam_score = make_gradcam(
                        model, image, image_transform(img_size), class_index=idx
                    )
                with left:
                    st.image(
                        overlay,
                        caption=("Grad-CAM: области, повлиявшие на выбранный класс "
                                 f"({DISPLAY.get(classes[cam_class], classes[cam_class])}, оценка {cam_score:.3f})"),
                        use_container_width=True,
                    )
                    st.caption("Тепловая карта является диагностической визуализацией, а не доказательством физически значимых признаков.")
        except Exception as e:
            st.error(f"Не удалось загрузить модель или обработать изображение: {e}")

# =====================================================================
elif page == "Пакетный анализ":
    st.subheader("Пакетный анализ астрофотографий")
    st.write("Загрузите несколько снимков для одинаковой обработки и выгрузки результатов в CSV.")
    if not MODEL_PATH.exists():
        st.info("Сначала обучите модель на странице «Обучение модели».")
    else:
        files = st.file_uploader("Изображения для анализа", type=["jpg", "jpeg", "png", "tif", "tiff", "webp"], accept_multiple_files=True)
        tta_batch = st.checkbox("Использовать TTA для каждого снимка (медленнее)", value=False, key="batch_tta")
        if files:
            st.caption(f"Выбрано файлов: {len(files)}. Ограничение: 100 файлов и 20 МБ на файл.")
            if len(files) > 100:
                st.error("За один запуск можно обработать не более 100 изображений.")
            elif any(f.size > 20 * 1024 * 1024 for f in files):
                st.error("Один или несколько файлов превышают 20 МБ.")
            elif st.button("Выполнить пакетную классификацию", type="primary"):
                model, classes, threshold, _, _, img_size = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
                from io import BytesIO
                inputs = []
                for uploaded_file in files:
                    try:
                        inputs.append((uploaded_file.name, Image.open(BytesIO(uploaded_file.getvalue())).copy()))
                    except Exception:
                        # Повреждённый файл попадёт в общий отчёт как отдельная ошибка.
                        inputs.append((uploaded_file.name, None))
                results = predict_many(model, inputs, classes, img_size, threshold, tta_batch)
                frame = pd.DataFrame(results)
                st.dataframe(frame, use_container_width=True, hide_index=True)
                ok_count = int((frame["status"] != "error").sum())
                low_count = int((frame["status"] == "not_supported").sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Обработано", f"{ok_count}/{len(frame)}")
                c2.metric("Отклонено", low_count)
                c3.metric("Ошибки файлов", int((frame["status"] == "error").sum()))
                st.download_button("Скачать результаты CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="astrovision_batch_predictions.csv", mime="text/csv")
                st.caption("Статус «not_supported» означает, что система отклонила снимок как неподходящий или недостаточно уверенно распознанный.")

# =====================================================================
elif page == "Обучение модели":
    st.subheader("Обучение модели")
    st.info("Для строгого отказа от посторонних изображений обязателен третий обучающий класс dataset/*/other. Добавляйте туда разнообразные снимки звёзд, планет, Луны, пейзажей, людей, схем и обычных фотографий.")
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
            counts_train, counts_val, counts_test = split_counts("train"), split_counts("val"), split_counts("test")
            ok = (set(counts_train) == {"galaxies", "nebulae"} and
                  set(counts_val) == {"galaxies", "nebulae"} and
                  set(counts_test) == {"galaxies", "nebulae"} and
                  all(counts_train.values()) and all(counts_val.values()) and all(counts_test.values()))
            if not ok:
                st.warning("Для исследовательского обучения нужны оба класса в dataset/train, dataset/val и dataset/test. См. README.md.")
            if st.button("🔎 Проверить датасет для конференции", disabled=not ok):
                audit_cmd = [sys.executable, str(ROOT / "audit_dataset.py"),
                             "--data_dir", str(ROOT / "dataset"),
                             "--require_metadata", "--require_provenance",
                             "--out", str(ROOT / "reports" / "dataset_audit.json")]
                audit = subprocess.run(audit_cmd, cwd=ROOT, capture_output=True, text=True)
                st.session_state["audit_passed"] = audit.returncode == 0
                st.session_state["audit_output"] = (audit.stdout or "") + (audit.stderr or "")
            if st.session_state.get("audit_output"):
                if st.session_state.get("audit_passed"):
                    st.success("Аудит пройден. Датасет прошёл проверки структуры, дубликатов, object_id и provenance.")
                else:
                    st.error("Аудит не пройден. Исправьте перечисленные ошибки.")
                st.code(st.session_state["audit_output"], language=None)
        with c2:
            st.markdown("**Вычислительное устройство**")
            if device.type == "cuda":
                st.success(f"🚀 GPU: {torch.cuda.get_device_name(0)} — обучение с AMP (float16)")
            else:
                st.info("💻 CPU: обучение возможно, но может быть медленным. Mixed precision AMP отключён на CPU.")
            st.caption("Скорость обучения и инференса зависит от устройства, размера модели и batch size.")

        st.markdown("**Гиперпараметры**")
        c1, c2, c3 = st.columns(3)
        epochs = c1.slider("Эпохи", 2, 60, 15)
        batch_size = c2.select_slider("Batch size", options=[4, 8, 16, 32], value=16)
        lr = c3.select_slider("Learning rate", options=[1e-5, 3e-5, 1e-4, 3e-4], value=1e-4, format_func=lambda v: f"{v:.0e}")
        architecture = st.selectbox("Архитектура нейросети", options=list(ARCHITECTURES),
                                    format_func=lambda x: {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet50", "convnext_tiny": "ConvNeXt-Tiny"}[x])
        c1, c2, c3 = st.columns(3)
        freeze_epochs = c1.slider("Эпох с замороженным backbone", 0, 5, 2)
        patience = c2.slider("Ранняя остановка (эпох без улучшения)", 0, 10, 4)
        threshold = c3.slider("Порог предупреждения о низкой уверенности", 0.3, 0.95, 0.65, 0.05)
        with st.expander("Дополнительно"):
            c1, c2, c3 = st.columns(3)
            use_amp = c1.checkbox("AMP (ускорение)", value=True)
            use_aug = c2.checkbox("Расширенные аугментации", value=True)
            workers = c3.number_input("Workers (0 — безопаснее на Windows)", 0, 8, 0)
            img_size = c1.select_slider("Размер изображения", options=[160, 224, 256], value=224)
            seed = c2.number_input("Seed (для воспроизводимости)", min_value=0, max_value=999999, value=42, step=1)
            weighted_loss = c3.checkbox("Взвешивать классы при дисбалансе", value=False)

        if st.button("🚀 Запустить обучение", type="primary", disabled=not ok):
            audit_cmd = [sys.executable, str(ROOT / "audit_dataset.py"),
                         "--data_dir", str(ROOT / "dataset"),
                         "--require_metadata", "--require_provenance",
                         "--out", str(ROOT / "reports" / "dataset_audit.json")]
            audit = subprocess.run(audit_cmd, cwd=ROOT, capture_output=True, text=True)
            st.session_state["audit_passed"] = audit.returncode == 0
            st.session_state["audit_output"] = (audit.stdout or "") + (audit.stderr or "")
            if audit.returncode != 0:
                st.error("Обучение не запущено: датасет не прошёл аудит.")
                st.code(st.session_state["audit_output"], language=None)
                st.stop()
            cmd = [sys.executable, str(ROOT / "train.py"), "--data_dir", str(ROOT / "dataset"),
                   "--epochs", str(epochs), "--batch_size", str(batch_size), "--lr", str(lr),
                   "--threshold", str(threshold), "--arch", architecture, "--freeze_epochs", str(freeze_epochs),
                   "--patience", str(patience), "--workers", str(workers),
                   "--img_size", str(img_size), "--seed", str(seed), "--progress_file", str(PROGRESS_PATH),
                   "--stop_file", str(STOP_FLAG), "--out", str(MODEL_PATH),
                   "--require_metadata", "--require_provenance"]
            if not use_amp:
                cmd.append("--no_amp")
            if not use_aug:
                cmd.append("--no_aug")
            if weighted_loss:
                cmd.append("--class_weighted_loss")
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
        c1.metric("Производительность", f"{1000/max(ms, 1e-6):.1f} изобр/с")
        c2.metric("Размер входа", f"{img_size}×{img_size}")
        c3.metric("Устройство бенчмарка", "CPU")
        st.caption("Бенчмарк выполняется на CPU, один снимок за раз; сравнивайте времена только при одинаковом оборудовании и настройках.")
        st.caption("Метрики рассчитаны скриптом обучения на папке test, если она была предоставлена.")
        if history:
            fig = px.line(pd.DataFrame(history).set_index("epoch")[["train_loss", "val_loss", "train_acc", "val_acc"]],
                          labels={"value": "Значение", "epoch": "Эпоха"}, title="История обучения")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        if metrics:
            cols = st.columns(4)
            for col, key, label in zip(cols, ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc"],
                                       ["Accuracy", "Balanced accuracy", "F1-score (macro)", "ROC-AUC"]):
                value = metrics.get(key)
                col.metric(label, f"{value:.3f}" if value is not None else "—")
            extra_cols = st.columns(3)
            for col, key, label in zip(extra_cols, ["matthews_corrcoef", "cohen_kappa", "ece_15_bins"],
                                       ["MCC", "Cohen’s kappa", "ECE (15 bins)"]):
                value = metrics.get(key)
                col.metric(label, f"{value:.3f}" if value is not None else "—")
            ci = metrics.get("bootstrap_95ci", {})
            if ci:
                st.markdown("**95% доверительные интервалы (bootstrap на тестовой выборке)**")
                ci_rows = [{"Метрика": key, "Нижняя граница": bounds[0], "Верхняя граница": bounds[1]}
                           for key, bounds in ci.items()
                           if isinstance(bounds, list) and len(bounds) == 2]
                if ci_rows:
                    st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)
                unit = ci.get("bootstrap_unit", "unknown")
                count = ci.get("n_bootstrap", "—")
                st.caption(f"Единица выборки: {unit}; повторов: {count}.")
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
        st.caption(f"Порог отказа по низкой уверенности: {threshold:.2f}. Дополнительно используется обучаемый класс other; абсолютной гарантии отказа для неизвестных данных нет.")

# =====================================================================
else:
    st.subheader("О проекте")
    st.markdown("""
**AstroVision AI v2.0** — учебно-исследовательская система, которая принимает только галактики и туманности. Внутренний класс `other` нужен для отказа от всех неподходящих изображений.

**Метод:** сравнительное обучение EfficientNet-B0, ResNet50 и ConvNeXt-Tiny (ImageNet), три обучаемых класса (`galaxies`, `nebulae`, `other`) и отказ от низкоуверенных результатов. Двухэтапное дообучение: сначала замороженный backbone и обучаемый классификатор, затем разморозка с малым learning rate.

**Воспроизводимость и оценка (v2.0):**
- фиксируемый seed и одинаковые настройки для повторных запусков;
- проверка точных дубликатов между train/val/test и группировки по object_id;
- единая letterbox-предобработка без искажения пропорций для train/validation/test/inference;
- проверка происхождения и лицензий изображений перед запуском из интерфейса;
- обязательный независимый test-набор для исследовательского обучения;
- Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, MCC, Cohen’s kappa;
- 95% bootstrap-интервалы и CSV с предсказаниями на каждом тестовом изображении.

**Инструменты анализа:**
- три архитектуры доступны через CLI `train.py --arch`; checkpoint хранит архитектуру и загружается автоматически;
- пакетная классификация до 100 снимков с CSV-отчётом и изоляцией ошибок отдельных файлов;
- Grad-CAM для визуальной проверки областей изображения, влияющих на решение модели;
- классический baseline на цветовых гистограммах, миниатюре яркости и логистической регрессии;
- балансируемое групповое разбиение с поиском кандидата, близкого к 70/15/15.

**Скорость:**
- Mixed precision (AMP, float16) на CUDA GPU; на CPU AMP отключён;
- `channels_last`, `pin_memory`, `non_blocking`; `cudnn.benchmark` выключен ради воспроизводимости;
- косинусное расписание learning rate с ненулевым шагом при размораживании backbone;
- замер скорости обучения (изображений/с) и бенчмарк инференса;
- опциональный TTA (test-time augmentation) — усреднение предсказаний с отражённым снимком.

**Обучение:**
- запуск обучения прямо из веб-интерфейса с живым прогрессом и графиком потерь;
- мягкая остановка после текущей эпохи; ранняя остановка по val_loss;
- умеренные аугментации (повороты, отражения, кропы и небольшие цветовые сдвиги);
- per-class метрики и история обучения в чекпоинте.

**Ограничения:** внутренний класс `other` и порог уверенности помогают отклонять неподходящие снимки, но не дают абсолютной гарантии для любого неизвестного изображения. Качество отказа зависит от разнообразия класса `other`; необходим отдельный OOD-тест. Softmax — оценка модели, а не вероятность истинности.

**Технологии:** Python, PyTorch, torchvision, Streamlit, Plotly, scikit-learn.

См. README.md: установка, структура датасета и публикация на Streamlit Community Cloud.
""")
