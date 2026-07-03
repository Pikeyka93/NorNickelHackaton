# ============================================================
# ВЛАДЕЛЕЦ: P4 (UI + отчёты + оформление под демо)
# TODO(P4):
#   - причесать под видео-демо (заголовок, примеры, подписи);
#   - режим экспертной проверки (отметить ошибочный тальк) — опционально;
#   - следить за стабильностью на реальных снимках (analyze уже в try/except).
# ВАЖНО: это ЖИВОЙ деплой (Streamlit + Cloudflare-туннель). Не ломать импорт/запуск.
#
# ПРИМЕЧАНИЕ: визуальный редизайн (v2) предложен P1 (Timur) поверх готового каркаса —
# контракт analyze() и порядок вызовов не менялись, только вёрстка/CSS/раскладка.
# Тема оформления — в .streamlit/config.toml. Согласовать с P4 при доп. правках.
# ============================================================
"""
Streamlit UI: загрузка -> вердикт сорта + синяя маска талька (с зумом) + %талька +
строка согласованности + экспорт CSV/PDF. Срастания НЕ показываем (вне задачи).

Run:  streamlit run app/main.py
Вызывает core.analyze() в процессе (без отдельного API).
"""
from __future__ import annotations

import base64
import math
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
_VERDICT_ICON = {"ordinary": "⛰️", "hard_to_process": "⛏️", "talc": "\U0001F9F4"}
_LEGEND = '<span style="color:#1656c9;font-size:1.15em;">■</span> тальк (синяя маска)'

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --nn-navy: #17284d;
    --nn-blue: #1656c9;
    --nn-blue-soft: #eaf1ff;
    --nn-border: #e7eaf1;
}

html, body, [class*="css"] { font-family: 'Manrope', -apple-system, sans-serif !important; }

/* убрать дефолтный тулбар/футер Streamlit, чтобы не выглядело как дев-стенд */
#MainMenu, footer, div[data-testid="stToolbar"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1220px; }

/* фон вместо плоского белого/серого — мягкие цветные пятна на светлой подложке */
[data-testid="stAppViewContainer"], .stApp {
    background:
        radial-gradient(circle at 10% -6%, rgba(22,86,201,.13), transparent 34%),
        radial-gradient(circle at 92% 8%, rgba(23,40,77,.10), transparent 30%),
        radial-gradient(circle at 78% 92%, rgba(22,86,201,.09), transparent 34%),
        radial-gradient(circle at 4% 88%, rgba(90,180,190,.10), transparent 32%),
        linear-gradient(180deg, #f2f5fb 0%, #eef2f9 100%);
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8f9fd 100%);
    box-shadow: 2px 0 16px rgba(23,40,77,.05);
}

/* --- зона загрузки файлов --- */
.snb-upload-label {
    display:flex; align-items:center; gap:.5rem; font-weight:700; color: var(--nn-navy);
    margin-bottom: .5rem; font-size: .98rem;
}
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #eef4ff 0%, #f8fafe 100%) !important;
    border: 1.5px dashed #a9c3ef !important;
    border-radius: 16px !important;
    transition: border-color .18s ease, background .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--nn-blue) !important;
    background: linear-gradient(135deg, #e2ecff 0%, #f4f8ff 100%) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { fill: var(--nn-blue) !important; }
[data-testid="stFileUploader"] section button {
    border-radius: 10px !important;
}

/* --- topbar --- */
.snb-topbar { display:flex; align-items:center; gap:.6rem; margin-bottom: .9rem; }
.snb-topbar .logo {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, var(--nn-blue), var(--nn-navy));
    display:flex; align-items:center; justify-content:center; font-size:1.15rem;
    box-shadow: 0 3px 8px rgba(22,86,201,.35);
}
.snb-topbar .brand { font-weight: 800; letter-spacing:.02em; color: var(--nn-navy); font-size: .95rem; }
.snb-topbar .tag {
    margin-left:auto; font-size:.72rem; font-weight:700; letter-spacing:.06em;
    color: var(--nn-blue); background: var(--nn-blue-soft); padding:.25rem .7rem;
    border-radius: 999px; text-transform: uppercase;
}

/* --- hero --- */
.snb-hero {
    position: relative; overflow: hidden;
    background: radial-gradient(circle at 88% -10%, rgba(255,255,255,.16), transparent 55%),
                linear-gradient(120deg, var(--nn-navy) 0%, var(--nn-blue) 100%);
    color: #fff;
    padding: 2rem 2.2rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 26px rgba(23, 40, 77, 0.28);
}
.snb-hero h1 { margin: 0 0 .45rem 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -.01em; }
.snb-hero p { margin: 0; opacity: .92; font-size: 1rem; max-width: 62ch; line-height: 1.5; }
.snb-hero .pills { margin-top: 1rem; display:flex; gap:.5rem; flex-wrap: wrap; }
.snb-hero .pill {
    font-size: .78rem; font-weight: 600; padding: .3rem .7rem; border-radius: 999px;
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
}

/* --- result card --- */
.snb-card {
    border: 1px solid var(--nn-border);
    border-radius: 16px;
    padding: 1.3rem 1.5rem 1.1rem;
    background: #ffffff;
    box-shadow: 0 3px 14px rgba(20, 30, 60, 0.06);
    margin-bottom: 1.1rem;
}
.snb-filename {
    font-weight: 700; color: var(--nn-navy); font-size: 1.02rem;
    display:flex; align-items:center; gap:.45rem; margin-bottom: .6rem;
}

/* verdict pill */
.snb-badge {
    display: inline-flex; align-items:center; gap:.4rem;
    padding: .4rem 1rem;
    border-radius: 999px;
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: .01em;
}

/* icon stat chip */
.snb-chipstat {
    display:flex; align-items:center; gap:.6rem;
    border-radius: 12px; padding: .55rem .8rem;
    background: #f7f8fc; border: 1px solid var(--nn-border);
    margin-bottom: .5rem;
}
.snb-chipstat .ico {
    width: 30px; height: 30px; border-radius: 8px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; font-size: .95rem;
    background: var(--nn-blue-soft);
}
.snb-chipstat .v { font-weight: 800; color: var(--nn-navy); font-size: 1.02rem; line-height:1.1; }
.snb-chipstat .l { font-size: .74rem; color: #6b7280; }

/* sidebar swatches */
.snb-swatch {
    display:flex; gap:.65rem; padding:.6rem .1rem; border-bottom: 1px solid #eef0f5;
}
.snb-swatch:last-child { border-bottom: none; }
.snb-swatch .ico {
    width: 32px; height:32px; border-radius:9px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; font-size:1rem;
}
.snb-swatch b { color: var(--nn-navy); }
.snb-swatch span.d { font-size: .82rem; color:#5a6472; }

/* stepper */
.snb-step { display:flex; gap:.65rem; align-items:flex-start; margin-bottom: .85rem; }
.snb-step .n {
    width: 22px; height:22px; border-radius:50%; background: var(--nn-blue); color:#fff;
    font-size:.72rem; font-weight:800; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; margin-top:.1rem;
}
.snb-step .t { font-size:.85rem; color:#42485a; line-height:1.4; }

/* overview chips row (batch) */
.snb-overview { display:flex; gap:.6rem; flex-wrap: wrap; margin: .3rem 0 1rem; }
.snb-ochip {
    display:flex; align-items:center; gap:.45rem; border-radius: 999px;
    padding: .4rem .85rem; font-weight:700; font-size:.86rem; border:1px solid var(--nn-border);
}

/* consistency banners look native to the theme */
div[data-testid="stAlertContentSuccess"], div[data-testid="stAlertContentWarning"] { font-weight: 600; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Помощники: декодирование / зум
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


# --------------------------------------------------------------------------- #
# Помощники: визуальные компоненты (HTML/SVG строятся вручную — без доп.
# зависимостей и внешних иконок, только CSS + встроенный SVG)
# --------------------------------------------------------------------------- #
def _verdict_badge(verdict: str) -> str:
    color = _VERDICT_COLOR.get(verdict, "#333")
    bg = _VERDICT_BG.get(verdict, "#eee")
    ru = C.CLASS_RU.get(verdict, verdict)
    icon = _VERDICT_ICON.get(verdict, "")
    return (f'<span class="snb-badge" style="color:{color};background:{bg};">'
            f'{icon} {ru.upper()}</span>')


def _chip_stat(icon: str, value: str, label: str) -> str:
    return (f'<div class="snb-chipstat"><div class="ico">{icon}</div>'
            f'<div><div class="v">{value}</div><div class="l">{label}</div></div></div>')


def _talc_ring_svg(pct: float, size: int = 132, stroke: int = 12) -> str:
    """Кольцевой (donut) индикатор доли талька — SVG, без внешних библиотек."""
    thr = C.TALC_VERDICT_THRESHOLD_PCT
    color = "#c23b22" if pct >= thr else "#1656c9"
    r = size / 2 - stroke
    circumference = 2 * math.pi * r
    frac = max(0.0, min(100.0, pct)) / 100.0
    dash = circumference * frac
    cx = cy = size / 2
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#eef0f5" stroke-width="{stroke}"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
            stroke-linecap="round" stroke-dasharray="{dash:.1f} {circumference:.1f}"
            transform="rotate(-90 {cx} {cy})"/>
        <text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="21" font-weight="800"
            fill="#17284d" font-family="Manrope, sans-serif">{pct:.1f}%</text>
        <text x="{cx}" y="{cy + 18}" text-anchor="middle" font-size="10.5" fill="#8993a6"
            font-family="Manrope, sans-serif">тальк</text>
    </svg>
    """


def _is_ok(consistency: str) -> bool:
    return "проверить" not in (consistency or "")


# --------------------------------------------------------------------------- #
# Сайдбар: глоссарий сортов + как это работает
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div class="snb-topbar"><div class="logo">\U0001FAA8</div>'
        '<div class="brand">СКАЖИ МНЕ, КТО ТВОЙ ШЛИФ</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### \U0001F4D8 Три сорта руды")
    _GLOSSARY = {
        "ordinary": ("обычная руда, металл извлекается стандартно"),
        "hard_to_process": ("тонкие срастания — металл извлечь сложнее"),
        "talc": ("много талька — мешает переработке"),
    }
    swatches = []
    for cls, desc in _GLOSSARY.items():
        color = _VERDICT_COLOR[cls]
        bg = _VERDICT_BG[cls]
        icon = _VERDICT_ICON[cls]
        ru = C.CLASS_RU[cls]
        swatches.append(
            f'<div class="snb-swatch"><div class="ico" style="background:{bg};">{icon}</div>'
            f'<div><b style="color:{color};">{ru}</b><br><span class="d">{desc}</span></div></div>'
        )
    st.markdown("".join(swatches), unsafe_allow_html=True)

    st.markdown("#### ⚙️ Как это работает")
    st.markdown(
        '<div class="snb-step"><div class="n">1</div>'
        '<div class="t">Классификатор (нейросеть) смотрит на снимок целиком и '
        'определяет сорт.</div></div>'
        '<div class="snb-step"><div class="n">2</div>'
        '<div class="t">Отдельный CV-алгоритм находит тальк на снимке и '
        'закрашивает его синим.</div></div>'
        '<div class="snb-step"><div class="n">3</div>'
        '<div class="t">Геолог смотрит результат и подтверждает или '
        'поправляет вердикт.</div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        f"Порог согласованности: тальк ≥ {C.TALC_VERDICT_THRESHOLD_PCT:.0f}% → сорт "
        f"«{C.CLASS_RU[C.CLASS_TALC]}». Панорамы обрабатываются тайлами с перекрытием."
    )

# --------------------------------------------------------------------------- #
# Topbar + Hero
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="snb-topbar"><div class="logo">\U0001FAA8</div>'
    '<div class="brand">NORNICKEL AI SCIENCE HACK</div>'
    '<div class="tag">MVP · демо</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="snb-hero">
        <h1>Скажи мне, кто твой шлиф</h1>
        <p>Загрузите фото среза руды (аншлиф) — программа определит сорт и покажет,
        где на снимке тальк, вместо ручной оценки геологом «на глаз».</p>
        <div class="pills">
            <span class="pill">\U0001F9E0 Классификатор сорта</span>
            <span class="pill">\U0001F535 CV-сегментация талька</span>
            <span class="pill">\U0001F5FA️ Поддержка панорам</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="snb-upload-label">\U0001F4E4 Загрузите снимок(и) аншлифа / '
    'панораму (TIFF / PNG / JPEG)</div>',
    unsafe_allow_html=True,
)
files = st.file_uploader(
    "Загрузите снимок(и) аншлифа / панораму (TIFF / PNG / JPEG)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not files:
    st.info("⬆️ Загрузите один или несколько снимков, чтобы начать анализ.")
    st.stop()

all_results = []
for file in files:
    data = file.getvalue()

    st.markdown('<div class="snb-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="snb-filename">\U0001F4CE {file.name}</div>',
                unsafe_allow_html=True)

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

    # --- верхняя строка: кольцо талька | вердикт+согласованность | быстрые чипы ---
    top_gauge, top_verdict, top_chips = st.columns([1.1, 2.4, 1.5])
    with top_gauge:
        st.markdown(_talc_ring_svg(talc_pct), unsafe_allow_html=True)
    with top_verdict:
        st.markdown(_verdict_badge(verdict), unsafe_allow_html=True)
        st.write(report.verdict_text(result))
        ok = _is_ok(result.get("consistency_check", ""))
        (st.success if ok else st.warning)(
            ("✅ " if ok else "⚠️ ") + result.get("consistency_check", ""))
    with top_chips:
        conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
        st.markdown(_chip_stat("\U0001F3AF", conf_str, "Уверенность классификатора"),
                    unsafe_allow_html=True)
        st.markdown(_chip_stat("⏱️", f"{result.get('processing_time_sec', 0):.2f} с",
                                "Время обработки"), unsafe_allow_html=True)

    st.write("")
    col_img, col_side = st.columns([3, 2])

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

    with col_side:
        with st.expander("\U0001F50E Технические детали", expanded=False):
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
                               mime="application/pdf", key=f"pdf_{file.name}",
                               use_container_width=True)
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

    # overview-чипы: сколько снимков какого сорта
    overview = []
    for cls in C.CLASSES:
        n = sum(1 for r in all_results if r.get("verdict") == cls)
        if n == 0:
            continue
        color = _VERDICT_COLOR[cls]
        bg = _VERDICT_BG[cls]
        icon = _VERDICT_ICON[cls]
        overview.append(
            f'<span class="snb-ochip" style="color:{color};background:{bg};">'
            f'{icon} {n} × {C.CLASS_RU[cls]}</span>'
        )
    st.markdown(f'<div class="snb-overview">{"".join(overview)}</div>', unsafe_allow_html=True)

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
                f"<td style='padding:.5rem .7rem;border-radius:8px 0 0 8px;'>{r.get('image_id', '')}</td>"
                f"<td style='padding:.5rem .7rem;color:{color};font-weight:700;'>{ru}</td>"
                f"<td style='padding:.5rem .7rem;'>{r.get('talc_pct', 0.0):.1f}%</td>"
                f"<td style='padding:.5rem .7rem;'>{conf_str}</td>"
                f"<td style='padding:.5rem .7rem;border-radius:0 8px 8px 0;'>{r.get('consistency_check', '')}</td>"
                f"</tr>"
            )
        table_html = (
            "<table style='width:100%;border-collapse:separate;border-spacing:0 6px;font-size:.9rem;'>"
            "<thead><tr style='text-align:left;color:#6b7280;font-size:.78rem;"
            "text-transform:uppercase;letter-spacing:.03em;'>"
            "<th style='padding:0 .7rem .3rem;'>Файл</th><th style='padding:0 .7rem .3rem;'>Сорт</th>"
            "<th style='padding:0 .7rem .3rem;'>Тальк, %</th><th style='padding:0 .7rem .3rem;'>Уверенность</th>"
            "<th style='padding:0 .7rem .3rem;'>Согласованность</th></tr></thead>"
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
