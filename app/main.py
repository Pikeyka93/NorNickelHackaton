"""
Streamlit UI: upload -> analyze -> verdict + phase mask (with zoom) + metrics + export.

Run:  streamlit run app/main.py
Calls core.analyze() in-process (no API needed). Set an API URL below if you'd rather
hit the FastAPI service instead.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

# streamlit puts app/ on sys.path, not the repo root — make config/core importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

import config as C
from core import analyze as analyze_mod
from core import report

st.set_page_config(page_title="Скажи мне, кто твой шлиф", layout="wide")

_LEGEND = (
    '<span style="color:#00c800">■</span> обычные срастания &nbsp;&nbsp;'
    '<span style="color:#dc0000">■</span> тонкие срастания &nbsp;&nbsp;'
    '<span style="color:#0000dc">■</span> тальк'
)


def _decode_original(data: bytes) -> np.ndarray:
    """RGB array at the same resolution analyze() used (so masks line up)."""
    import cv2
    from core import segment
    bgr = segment.read_image_bgr(data)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _composite(original_rgb: np.ndarray, mask_b64: str) -> np.ndarray:
    """Alpha-blend the RGBA phase mask (base64 PNG) over the original."""
    if not mask_b64:
        return original_rgb
    import cv2
    buf = np.frombuffer(base64.b64decode(mask_b64), np.uint8)
    bgra = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.shape[2] < 4:
        return original_rgb
    mask_rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = (bgra[:, :, 3:4].astype(np.float32) / 255.0) * C.MASK_ALPHA
    out = original_rgb.astype(np.float32) * (1 - alpha) + mask_rgb * alpha
    return out.clip(0, 255).astype(np.uint8)


def _zoom(img: np.ndarray, factor: float, cx: float, cy: float) -> np.ndarray:
    """Crop a centered region for a simple zoom (factor 1 = full image)."""
    if factor <= 1.0:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h / factor), int(w / factor)
    y0 = int(np.clip(cy * h - ch / 2, 0, h - ch))
    x0 = int(np.clip(cx * w - cw / 2, 0, w - cw))
    return img[y0:y0 + ch, x0:x0 + cw]


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("Скажи мне, кто твой шлиф")
st.caption("Классификация геолого-технологического сорта руды по аншлифу (OM).")

files = st.file_uploader(
    "Загрузите снимок(и) аншлифа (TIFF / PNG / JPEG)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True,
)

if not files:
    st.info("Загрузите изображение, чтобы начать анализ.")
    st.stop()

all_results = []
for file in files:
    data = file.getvalue()
    with st.spinner(f"Анализ {file.name}…"):
        result = analyze_mod.analyze(data, image_id=file.name)
    all_results.append(result)

    st.divider()
    st.subheader(file.name)

    verdict_ru = C.CLASS_RU.get(result["verdict"], result["verdict"])
    color = {"ordinary": "#00a000", "hard_to_process": "#c00000", "talc": "#0000c0"}.get(result["verdict"], "#333")
    st.markdown(f"### Вердикт: <span style='color:{color}'>{verdict_ru}</span>", unsafe_allow_html=True)
    st.write(report.verdict_text(result))
    st.caption(f"Источник вердикта: {result.get('verdict_source', '?')} · "
               f"время: {result['processing_time_sec']} с")

    col_img, col_metrics = st.columns([3, 2])

    with col_img:
        st.markdown("**Маска фаз** &nbsp; " + _LEGEND, unsafe_allow_html=True)
        try:
            original = _decode_original(data)
            overlay = _composite(original, result["mask_png_b64"])
            z = st.slider("Зум", 1.0, 6.0, 1.0, 0.5, key=f"z_{file.name}")
            if z > 1.0:
                c1, c2 = st.columns(2)
                cx = c1.slider("центр X", 0.0, 1.0, 0.5, 0.05, key=f"cx_{file.name}")
                cy = c2.slider("центр Y", 0.0, 1.0, 0.5, 0.05, key=f"cy_{file.name}")
            else:
                cx = cy = 0.5
            tab_ov, tab_orig = st.tabs(["С маской", "Оригинал"])
            tab_ov.image(_zoom(overlay, z, cx, cy), use_container_width=True)
            tab_orig.image(_zoom(original, z, cx, cy), use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось отрисовать маску: {e}")

    with col_metrics:
        m = result["metrics"]
        st.markdown("**Метрики (доля от площади шлифа)**")
        st.dataframe({
            "Метрика": ["Сульфиды", "Обычные срастания", "Тонкие срастания", "Тальк"],
            "%": [m["sulfide_area_pct"], m["ordinary_pct"], m["fine_pct"], m["talc_pct"]],
        }, hide_index=True, use_container_width=True)
        if result.get("classifier_probs"):
            st.markdown("**Вероятности классификатора**")
            st.dataframe({"класс": list(result["classifier_probs"].keys()),
                          "p": [round(v, 3) for v in result["classifier_probs"].values()]},
                         hide_index=True, use_container_width=True)

        # PDF download (per image)
        try:
            import cv2
            original = _decode_original(data)
            overlay = _composite(original, result["mask_png_b64"])
            orig_png = cv2.imencode(".png", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))[1].tobytes()
            ov_png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tobytes()
            pdf_path = report.build_pdf(result, original_png=orig_png, overlay_png=ov_png)
            st.download_button("📄 Скачать PDF-отчёт", data=open(pdf_path, "rb").read(),
                               file_name=f"{result['image_id']}.pdf", mime="application/pdf",
                               key=f"pdf_{file.name}")
        except Exception as e:
            st.caption(f"PDF недоступен: {e}")

# Batch CSV export
st.divider()
csv_text = report.results_to_csv(all_results)
st.download_button("⬇️ Скачать CSV (все изображения)", data=csv_text.encode("utf-8"),
                   file_name="analysis.csv", mime="text/csv")
