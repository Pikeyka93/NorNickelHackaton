# ============================================================
# ВЛАДЕЛЕЦ: P4 (UI + отчёты + оформление под демо)
# TODO(P4):
#   - причесать под видео-демо (заголовок, примеры, подписи);
#   - режим экспертной проверки (отметить ошибочный тальк) — опционально;
#   - следить за стабильностью на реальных снимках (analyze уже в try/except).
# ВАЖНО: это ЖИВОЙ деплой (Streamlit + Cloudflare-туннель). Не ломать импорт/запуск.
#
# ПРИМЕЧАНИЕ: визуальный редизайн предложен P1 (Timur) поверх готового каркаса —
# контракт analyze() и порядок вызовов не менялись, только вёрстка/CSS/раскладка.
# Смержено в src через PR, согласовать с P4 при доп. правках.
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
import pandas as pd
import streamlit as st

import config as C
from core import analyze as analyze_mod
from core import report

st.set_page_config(
    page_title="Скажи мне, кто твой шлиф",
    page_icon="\U0001FAA8",
    layout="wide",
    initial_sidebar_state="expanded",
)

_VERDICT_COLOR = {"ordinary": "#1a9e4b", "hard_to_process": "#c23b22", "talc": "#1656c9"}
_VERDICT_BG = {"ordinary": "#eaf7ee", "hard_to_process": "#fdecea", "talc": "#e9f0fd"}
_LEGEND = '<span style="color:#1656c9;font-size:1.1em;">■</span> тальк (синяя маска)'

_CSS = """
<style>
:root {
    --nn-navy: #17284d;
    --nn-blue: #1656c9;
}
.block-container { padding-top: 1.2rem; max-width: 1200px; }

/* hero banner */
.snb-hero {
    background: linear-gradient(120deg, var(--nn-navy) 0%, var(--nn-blue) 100%);
    color: #fff;
    padding: 1.6rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.4rem;
    box-shadow: 0 6px 18px rgba(23, 40, 77, 0.25);
}
.snb-hero h1 { margin: 0 0 .3rem 0; font-size: 1.7rem; }
.snb-hero p { margin: 0; opacity: .9; font-size: .98rem; }

/* result card */
.snb-card {
    border: 1px solid #e7e9ee;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    background: #ffffff;
    box-shadow: 0 2px 10px rgba(20, 30, 60, 0.05);
    margin-bottom: .8rem;
}

/* verdict pill */
.snb-badge {
    display: inline-block;
    padding: .32rem .9rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: .01em;
}

/* stat card */
.snb-stat {
    border-radius: 12px;
    padding: .7rem 1rem;
    background: #f7f8fb;
    border: 1px solid #eceef3;
    text-align: center;
}
.snb-stat .v { font-size: 1.6rem; font-weight: 700; color: var(--nn-navy); }
.snb-stat .l { font-size: .8rem; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }

/* talc gauge */
.snb-gauge-track { background: #eef0f5; border-radius: 999px; height: 10px; overflow: hidden; }
.snb-gauge-fill { height: 100%; border-radius: 999px; }

/* legend chips in sidebar */
.snb-chip { display: flex; align-items: center; gap: .5rem; margin-bottom: .5rem; }
.snb-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Помощники
# --------------------------------------------------------------------------- #
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


def _stat_card(value: str, label: str) -> str:
    return f'<div class="snb-stat"><div class="v">{value}</div><div class="l">{label}</div></div>'


def _verdict_badge(verdict: str) -> str:
    color = _VERDICT_COLOR.get(verdict, "#333")
    bg = _VERDICT_BG.get(verdict, "#eee")
    ru = C.CLASS_RU.get(verdict, verdict)
    return (f'<span class="snb-badge" style="color:{color};background:{bg};">'
            f'{ru.upper()}</span>')


def _talc_gauge(pct: float) -> str:
    thr = C.TALC_VERDICT_THRESHOLD_PCT
    color = "#c23b22" if pct >= thr else "#1656c9"
    width = max(2, min(100, pct))
    return (f'<div class="snb-gauge-track">'
            f'<div class="snb-gauge-fill" style="width:{width}%;background:{color};"></div>'
            f'</div>')


def _is_ok(consistency: str) -> bool:
    return "проверить" not in (consistency or "")


# --------------------------------------------------------------------------- #
# Сайдбар: глоссарий сортов + как это работает
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### \U0001F4D8 Три сорта руды")
    _GLOSSARY = {
        "ordinary": "обычная руда, металл извлекается стандартно",
        "hard_to_process": "тонкие срастания — металл извлечь сложнее",
        "talc": "много талька — мешает переработке",
    }
    for cls, desc in _GLOSSARY.items():
        color = _VERDICT_COLOR[cls]
        ru = C.CLASS_RU[cls]
        st.markdown(
            f'<div class="snb-chip"><span class="snb-dot" style="background:{color};"></span>'
            f'<div><b>{ru}</b><br><span style="font-size:.85rem;color:#555;">{desc}</span></div></div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown("### ⚙️ Как это работает")
    st.caption(
        "1. Классификатор (нейросеть) смотрит на снимок целиком и определяет сорт.\n\n"
        "2. Отдельный CV-алгоритм находит тальк на снимке и закрашивает его синим.\n\n"
        "3. Геолог смотрит результат и подтверждает или поправляет вердикт."
    )
    st.divider()
    st.caption(
        f"Порог согласованности: тальк ≥ {C.TALC_VERDICT_THRESHOLD_PCT:.0f}% → сорт "
        f"«{C.CLASS_RU[C.CLASS_TALC]}». Панорамы обрабатываются тайлами с перекрытием."
    )

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="snb-hero">
        <h1>\U0001FAA8 Скажи мне, кто твой шлиф</h1>
        <p>Загрузите фото среза руды (аншлиф) — программа определит сорт и покажет,
        где на снимке тальк. Классификатор сорта + классический CV для сегментации талька.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

files = st.file_uploader(
    "Загрузите снимок(и) аншлифа / панораму (TIFF / PNG / JPEG)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True,
)

if not files:
    st.info("⬆️ Загрузите один или несколько снимков, чтобы начать анализ.")
    st.stop()

all_results = []
for file in files:
    data = file.getvalue()

    with st.container(border=False):
        st.markdown('<div class="snb-card">', unsafe_allow_html=True)
        st.markdown(f"#### \U0001F4CE {file.name}")

        # analyze() в try/except: кривой файл не должен ронять всё приложение (живой деплой!)
        try:
            with st.spinner(f"Анализирую {file.name}…"):
                result = analyze_mod.analyze(data, image_id=file.name)
        except Exception as e:
            st.error(f"Не удалось обработать «{file.name}»: {e}. "
                     "Проверьте, что это корректное изображение аншлифа.")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        all_results.append(result)

        verdict = result.get("verdict", C.CLASS_ORDINARY)
        talc_pct = result.get("talc_pct", 0.0)
        conf = result.get("classifier_confidence")

        top_l, top_r = st.columns([2, 3])
        with top_l:
            st.markdown(_verdict_badge(verdict), unsafe_allow_html=True)
        with top_r:
            ok = _is_ok(result.get("consistency_check", ""))
            (st.success if ok else st.warning)(
                ("✅ " if ok else "⚠️ ") + result.get("consistency_check", ""))

        st.write(report.verdict_text(result))

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
                tab_ov, tab_orig = st.tabs(["\U0001F535 Тальк-маска", "\U0001F5BC️ Оригинал"])
                tab_ov.image(_zoom(overlay, z, cx, cy), use_container_width=True)
                tab_orig.image(_zoom(original, z, cx, cy), use_container_width=True)
            except Exception as e:
                st.warning(f"Не удалось отрисовать маску: {e}")

        with col_metrics:
            st.markdown("**Метрики**")
            m1, m2 = st.columns(2)
            m1.markdown(_stat_card(f"{talc_pct:.1f}%", "Доля талька"), unsafe_allow_html=True)
            conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
            m2.markdown(_stat_card(conf_str, "Уверенность"), unsafe_allow_html=True)
            st.caption(" ")
            st.markdown(_talc_gauge(talc_pct), unsafe_allow_html=True)
            st.caption(f"0% {'':<40} {C.TALC_VERDICT_THRESHOLD_PCT:.0f}% порог")

            with st.expander("\U0001F50E Технические детали"):
                st.write(f"ID образца: `{result.get('image_id', '')}`")
                st.write(f"Источник вердикта: `{result.get('verdict_source', '?')}`")
                st.write(f"Время обработки: {result.get('processing_time_sec', 0):.2f} с")

            try:
                import cv2
                original = _decode_original(data)
                overlay = _decode_png_b64(result.get("talc_mask_png_b64", ""))
                orig_png = cv2.imencode(".png", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))[1].tobytes()
                ov_png = (cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tobytes()
                          if overlay is not None else None)
                pdf_path = report.build_pdf(result, original_png=orig_png, talc_overlay_png=ov_png)
                st.download_button("\U0001F4C4 Скачать PDF-отчёт", data=open(pdf_path, "rb").read(),
                                   file_name=f"{result.get('image_id', 'report')}.pdf",
                                   mime="application/pdf", key=f"pdf_{file.name}")
            except Exception as e:
                st.caption(f"PDF недоступен: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Сводка по батчу
# --------------------------------------------------------------------------- #
if all_results:
    st.markdown("### \U0001F4CA Сводка по загруженным снимкам")

    df = pd.DataFrame([{
        "Файл": r.get("image_id", ""),
        "Сорт": C.CLASS_RU.get(r.get("verdict", ""), r.get("verdict", "")),
        "Тальк, %": r.get("talc_pct", 0.0),
        "Уверенность": r.get("classifier_confidence"),
        "Согласованность": r.get("consistency_check", ""),
    } for r in all_results])

    sum_l, sum_r = st.columns([3, 2])
    with sum_l:
        # Раскрашенная HTML-таблица вручную (без pandas Styler — тот тянет jinja2>=3.1,
        # которого может не быть на сервере; так надёжнее для живого деплоя).
        rows_html = []
        for r in all_results:
            verdict = r.get("verdict", "")
            color = _VERDICT_COLOR.get(verdict, "#333")
            bg = _VERDICT_BG.get(verdict, "#fff")
            ru = C.CLASS_RU.get(verdict, verdict)
            conf = r.get("classifier_confidence")
            conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
            rows_html.append(
                f"<tr style='background:{bg};'>"
                f"<td style='padding:.4rem .6rem;'>{r.get('image_id', '')}</td>"
                f"<td style='padding:.4rem .6rem;color:{color};font-weight:700;'>{ru}</td>"
                f"<td style='padding:.4rem .6rem;'>{r.get('talc_pct', 0.0):.1f}%</td>"
                f"<td style='padding:.4rem .6rem;'>{conf_str}</td>"
                f"<td style='padding:.4rem .6rem;'>{r.get('consistency_check', '')}</td>"
                f"</tr>"
            )
        table_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:.92rem;'>"
            "<thead><tr style='text-align:left;border-bottom:2px solid #e5e7eb;'>"
            "<th style='padding:.4rem .6rem;'>Файл</th><th style='padding:.4rem .6rem;'>Сорт</th>"
            "<th style='padding:.4rem .6rem;'>Тальк, %</th><th style='padding:.4rem .6rem;'>Уверенность</th>"
            "<th style='padding:.4rem .6rem;'>Согласованность</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)
    with sum_r:
        counts = df["Сорт"].value_counts()
        st.bar_chart(counts)

    st.download_button("⬇️ Скачать CSV (все изображения)",
                       data=report.results_to_csv(all_results).encode("utf-8"),
                       file_name="analysis.csv", mime="text/csv")

st.divider()
st.caption("Норникель AI Science Hack · «Скажи мне, кто твой шлиф» · "
           "сегментация талька — классический CV, сорт руды — классификатор.")
