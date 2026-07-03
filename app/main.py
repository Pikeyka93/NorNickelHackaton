# ============================================================
# ВЛАДЕЛЕЦ: P4 (UI + отчёты + оформление под демо)
# TODO(P4):
#   - причесать под видео-демо (заголовок, примеры, подписи);
#   - режим экспертной проверки (отметить ошибочный тальк) — опционально;
#   - следить за стабильностью на реальных снимках (analyze уже в try/except).
# ВАЖНО: это ЖИВОЙ деплой (Streamlit + Cloudflare-туннель). Не ломать импорт/запуск.
# ============================================================
"""
Streamlit UI: загрузка -> вердикт сорта + синяя маска талька (с зумом) + %талька +
строка согласованности + экспорт CSV/PDF. Срастания НЕ показываем (вне задачи).

Run:  streamlit run app/main.py
Вызывает core.analyze() в процессе (без отдельного API).
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

# streamlit кладёт app/ в sys.path, а не корень репо — делаем config/core импортируемыми
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

import config as C
from core import analyze as analyze_mod
from core import report

st.set_page_config(page_title="Скажи мне, кто твой шлиф", layout="wide")

_LEGEND = '<span style="color:#0000dc">■</span> тальк (синяя маска)'
_VERDICT_COLOR = {"ordinary": "#00a000", "hard_to_process": "#c00000", "talc": "#0000c0"}


def _decode_png_b64(b64: str) -> np.ndarray | None:
    """base64 PNG (BGR) -> RGB-массив."""
    if not b64:
        return None
    import cv2
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


def _decode_original(data: bytes) -> np.ndarray:
    """RGB-оригинал в том же разрешении, что и analyze()."""
    import cv2
    from core import segment
    return cv2.cvtColor(segment.read_image_bgr(data), cv2.COLOR_BGR2RGB)


def _zoom(img: np.ndarray, factor: float, cx: float, cy: float) -> np.ndarray:
    if factor <= 1.0 or img is None:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h / factor), int(w / factor)
    y0 = int(np.clip(cy * h - ch / 2, 0, h - ch))
    x0 = int(np.clip(cx * w - cw / 2, 0, w - cw))
    return img[y0:y0 + ch, x0:x0 + cw]


# --------------------------------------------------------------------------- #
st.title("Скажи мне, кто твой шлиф")
st.caption("Сорт руды (классификатор) + сегментация талька по аншлифу (OM). "
           "Панорама — рабочий вход инференса.")

files = st.file_uploader(
    "Загрузите снимок(и) аншлифа / панораму (TIFF / PNG / JPEG)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True,
)

if not files:
    st.info("Загрузите изображение, чтобы начать анализ.")
    st.stop()

all_results = []
for file in files:
    st.divider()
    st.subheader(file.name)

    data = file.getvalue()
    # analyze() в try/except: кривой файл не должен ронять всё приложение (живой деплой!)
    try:
        with st.spinner(f"Анализ {file.name}…"):
            result = analyze_mod.analyze(data, image_id=file.name)
    except Exception as e:
        st.error(f"Не удалось обработать «{file.name}»: {e}. "
                 "Проверьте, что это корректное изображение аншлифа.")
        continue

    all_results.append(result)

    verdict = result.get("verdict", C.CLASS_ORDINARY)
    verdict_ru = C.CLASS_RU.get(verdict, verdict)
    color = _VERDICT_COLOR.get(verdict, "#333")
    st.markdown(f"### Сорт руды: <span style='color:{color}'>{verdict_ru}</span>",
                unsafe_allow_html=True)
    st.write(report.verdict_text(result))
    conf = result.get("classifier_confidence")
    st.caption(f"Источник вердикта: {result.get('verdict_source', '?')} · "
               f"уверенность: {conf:.2f} · " if isinstance(conf, (int, float))
               else f"Источник вердикта: {result.get('verdict_source', '?')} · ")

    col_img, col_metrics = st.columns([3, 2])

    with col_img:
        st.markdown("**Маска талька** &nbsp; " + _LEGEND, unsafe_allow_html=True)
        try:
            overlay = _decode_png_b64(result.get("talc_mask_png_b64", ""))
            original = _decode_original(data)
            if overlay is None:
                overlay = original  # заглушка без сегментации
            z = st.slider("Зум", 1.0, 6.0, 1.0, 0.5, key=f"z_{file.name}")
            if z > 1.0:
                c1, c2 = st.columns(2)
                cx = c1.slider("центр X", 0.0, 1.0, 0.5, 0.05, key=f"cx_{file.name}")
                cy = c2.slider("центр Y", 0.0, 1.0, 0.5, 0.05, key=f"cy_{file.name}")
            else:
                cx = cy = 0.5
            tab_ov, tab_orig = st.tabs(["Тальк-маска", "Оригинал"])
            tab_ov.image(_zoom(overlay, z, cx, cy), use_container_width=True)
            tab_orig.image(_zoom(original, z, cx, cy), use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось отрисовать маску: {e}")

    with col_metrics:
        st.markdown("**Метрики**")
        st.metric("Доля талька, %", result.get("talc_pct", 0.0))
        st.info("🔎 " + result.get("consistency_check", ""))
        if isinstance(conf, (int, float)):
            st.metric("Уверенность классификатора", round(conf, 3))

        try:
            import cv2
            original = _decode_original(data)
            overlay = _decode_png_b64(result.get("talc_mask_png_b64", ""))
            orig_png = cv2.imencode(".png", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))[1].tobytes()
            ov_png = (cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tobytes()
                      if overlay is not None else None)
            pdf_path = report.build_pdf(result, original_png=orig_png, talc_overlay_png=ov_png)
            st.download_button("📄 Скачать PDF-отчёт", data=open(pdf_path, "rb").read(),
                               file_name=f"{result.get('image_id', 'report')}.pdf",
                               mime="application/pdf", key=f"pdf_{file.name}")
        except Exception as e:
            st.caption(f"PDF недоступен: {e}")

# Batch CSV export
if all_results:
    st.divider()
    st.download_button("⬇️ Скачать CSV (все изображения)",
                       data=report.results_to_csv(all_results).encode("utf-8"),
                       file_name="analysis.csv", mime="text/csv")
