# ============================================================
# ВЛАДЕЛЕЦ: P4 (UI + отчёты + оформление под демо)
# ВАЖНО: это живой Streamlit-контур. Контракт analyze() не ломать без P1/P2/P3.
# ============================================================
"""
Streamlit UI: загрузка снимка -> verdict сорта -> маска талька с зумом ->
таблица метрик -> экспертная проверка -> PDF/CSV/JSON экспорт.

Run: streamlit run app/main.py
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

# streamlit кладёт app/ в sys.path, а не корень репо — делаем config/core импортируемыми.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
from core import analyze as analyze_mod
from core import report

try:  # pandas есть в requirements, но UI должен деградировать дружелюбно.
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

Image.MAX_IMAGE_PIXELS = None

APP_TITLE = "Скажи мне, кто твой шлиф"
DEPLOY_URL = "https://plant-conceptual-closing-characteristics.trycloudflare.com"

CLASS_META = {
    C.CLASS_ORDINARY: {
        "title": "Рядовая",
        "description": "обычная руда, нормально обогащается",
        "color": "#18864b",
        "soft": "#eaf7ef",
    },
    C.CLASS_HARD: {
        "title": "Труднообогатимая",
        "description": "металл извлекается сложнее",
        "color": "#b25f00",
        "soft": "#fff3e1",
    },
    C.CLASS_TALC: {
        "title": "Оталькованная",
        "description": "много талька, он мешает переработке",
        "color": "#1f5fbf",
        "soft": "#eaf1ff",
    },
}


st.set_page_config(page_title=APP_TITLE, page_icon="🔬", layout="wide")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }
        .app-kicker {
            color: #536171;
            font-size: 0.95rem;
            margin-top: -0.5rem;
        }
        .verdict-panel {
            border: 1px solid #d8dee8;
            border-left-width: 7px;
            border-radius: 8px;
            padding: 16px 18px;
            margin: 0.4rem 0 0.8rem;
            background: #fbfcfe;
        }
        .verdict-label {
            color: #536171;
            font-size: 0.78rem;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .verdict-title {
            font-size: 1.85rem;
            font-weight: 720;
            line-height: 1.1;
            margin-top: 0.2rem;
        }
        .verdict-text {
            color: #2f3b49;
            margin-top: 0.45rem;
        }
        .status-badge {
            display: inline-block;
            border-radius: 999px;
            border: 1px solid #cdd5df;
            padding: 4px 10px;
            color: #39485a;
            background: #f6f8fb;
            font-size: 0.82rem;
            white-space: nowrap;
        }
        .status-badge.ok {
            color: #16613b;
            border-color: #b7dec8;
            background: #eaf7ef;
        }
        .status-badge.warn {
            color: #8a4b00;
            border-color: #f0c77e;
            background: #fff4dd;
        }
        .metric-note {
            color: #536171;
            font-size: 0.9rem;
        }
        div[data-testid="stMetric"] {
            background: #171c25;
            border: 1px solid #303745;
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 118px;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
            color: #a9b7c9 !important;
            font-weight: 560;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
            color: #f5f7fb !important;
            line-height: 1.08;
            overflow-wrap: anywhere;
        }
        .download-row button {
            min-height: 2.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _cached_analyze(data: bytes, image_id: str) -> dict[str, Any]:
    """Кэш нужен, чтобы слайдеры зума не запускали тяжелый analyze() заново."""
    return analyze_mod.analyze(data, image_id=image_id)


def _safe_filename(name: str, suffix: str) -> str:
    stem = Path(name or "report").name
    stem = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", stem).strip("._") or "report"
    if stem.lower().endswith(suffix.lower()):
        return stem
    return f"{stem}{suffix}"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_conf(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "—"


def _source_ru(source: Any) -> str:
    source = str(source or "unknown")
    if source == "classifier":
        return "классификатор P2"
    if source.startswith("fallback:talc"):
        return "fallback по доле талька"
    if source.startswith("fallback"):
        return "fallback без весов"
    if source.startswith("stub"):
        return "заглушка"
    return source


def _decode_uploaded(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("не удалось прочитать изображение") from exc
    return np.asarray(img)


def _decode_png_b64(b64: str | None) -> np.ndarray | None:
    if not b64:
        return None
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img)


def _resize_max_side(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    resized = Image.fromarray(img).resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )
    return np.asarray(resized)


def _array_to_png_bytes(img: np.ndarray | None, max_side: int = 1600) -> bytes | None:
    if img is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(_resize_max_side(img, max_side=max_side)).save(buf, format="PNG")
    return buf.getvalue()


def _array_to_data_url(img: np.ndarray, max_side: int = 2200) -> str:
    png = _array_to_png_bytes(img, max_side=max_side) or b""
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _viewer_id(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", key)


def _interactive_viewer(img: np.ndarray | None, key: str, height: int = 520) -> None:
    if img is None:
        return

    root_id = f"viewer_{_viewer_id(key)}"
    data_url = _array_to_data_url(img)
    components.html(
        f"""
        <div id="{root_id}" class="ore-viewer">
          <div class="ore-viewer__tools">
            <button type="button" data-action="out" title="Уменьшить">−</button>
            <button type="button" data-action="reset" title="Сбросить">↺</button>
            <button type="button" data-action="in" title="Увеличить">+</button>
          </div>
          <div class="ore-viewer__stage">
            <img src="{data_url}" alt="ore section" draggable="false" />
          </div>
        </div>
        <style>
          #{root_id} {{
            height: {height}px;
            width: 100%;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          #{root_id} .ore-viewer__tools {{
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            height: 34px;
            margin-bottom: 8px;
          }}
          #{root_id} button {{
            width: 34px;
            height: 34px;
            border: 1px solid #394150;
            border-radius: 7px;
            background: #171b24;
            color: #f5f7fb;
            font-size: 18px;
            line-height: 1;
            cursor: pointer;
          }}
          #{root_id} button:hover {{
            background: #202838;
            border-color: #5b6a82;
          }}
          #{root_id} .ore-viewer__stage {{
            position: relative;
            height: calc(100% - 42px);
            overflow: hidden;
            border: 1px solid #303745;
            border-radius: 8px;
            background: #0f1118;
          }}
          #{root_id} img {{
            position: absolute;
            top: 50%;
            left: 50%;
            max-width: 100%;
            max-height: 100%;
            transform-origin: center center;
            user-select: none;
            cursor: grab;
            will-change: transform;
          }}
          #{root_id}.is-dragging img {{
            cursor: grabbing;
          }}
        </style>
        <script>
          (() => {{
            const root = document.getElementById({json.dumps(root_id)});
            const stage = root.querySelector('.ore-viewer__stage');
            const img = root.querySelector('img');
            const tools = root.querySelector('.ore-viewer__tools');
            const state = {{ scale: 1, tx: 0, ty: 0, dragging: false, x: 0, y: 0 }};

            function apply() {{
              img.style.transform =
                `translate(-50%, -50%) translate(${{state.tx}}px, ${{state.ty}}px) scale(${{state.scale}})`;
            }}

            function zoomAt(nextScale, clientX, clientY) {{
              const oldScale = state.scale;
              nextScale = Math.max(1, Math.min(8, nextScale));
              if (nextScale === oldScale) return;
              const rect = stage.getBoundingClientRect();
              const localX = clientX - rect.left - rect.width / 2 - state.tx;
              const localY = clientY - rect.top - rect.height / 2 - state.ty;
              const ratio = nextScale / oldScale;
              state.tx -= localX * (ratio - 1);
              state.ty -= localY * (ratio - 1);
              state.scale = nextScale;
              apply();
            }}

            function reset() {{
              state.scale = 1;
              state.tx = 0;
              state.ty = 0;
              apply();
            }}

            stage.addEventListener('wheel', (event) => {{
              event.preventDefault();
              const factor = event.deltaY < 0 ? 1.14 : 0.88;
              zoomAt(state.scale * factor, event.clientX, event.clientY);
            }}, {{ passive: false }});

            stage.addEventListener('mousedown', (event) => {{
              if (event.button !== 0) return;
              state.dragging = true;
              state.x = event.clientX;
              state.y = event.clientY;
              root.classList.add('is-dragging');
            }});

            window.addEventListener('mousemove', (event) => {{
              if (!state.dragging) return;
              state.tx += event.clientX - state.x;
              state.ty += event.clientY - state.y;
              state.x = event.clientX;
              state.y = event.clientY;
              apply();
            }});

            window.addEventListener('mouseup', () => {{
              state.dragging = false;
              root.classList.remove('is-dragging');
            }});

            stage.addEventListener('dblclick', reset);

            tools.addEventListener('click', (event) => {{
              const action = event.target?.dataset?.action;
              if (action === 'in') {{
                const rect = stage.getBoundingClientRect();
                zoomAt(state.scale * 1.25, rect.left + rect.width / 2, rect.top + rect.height / 2);
              }}
              if (action === 'out') {{
                const rect = stage.getBoundingClientRect();
                zoomAt(state.scale / 1.25, rect.left + rect.width / 2, rect.top + rect.height / 2);
              }}
              if (action === 'reset') reset();
            }});

            img.addEventListener('load', apply);
            apply();
          }})();
        </script>
        """,
        height=height,
        scrolling=False,
    )


def _verdict_meta(verdict: str) -> dict[str, str]:
    return CLASS_META.get(verdict, {
        "title": C.CLASS_RU.get(verdict, verdict),
        "description": "нет описания",
        "color": "#39485a",
        "soft": "#f6f8fb",
    })


def _consistency_badge(result: dict[str, Any]) -> tuple[str, str]:
    text = str(result.get("consistency_check") or "согласованность не рассчитана")
    lowered = text.lower()
    tone = "warn" if "провер" in lowered or "но талька" in lowered else "ok"
    return tone, text


def _metric_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    verdict = result.get("verdict", "")
    return [
        {"Метрика": "Сорт руды", "Значение": C.CLASS_RU.get(verdict, verdict)},
        {"Метрика": "Доля талька", "Значение": _fmt_pct(result.get("talc_pct"))},
        {"Метрика": "Уверенность классификатора", "Значение": _fmt_conf(result.get("classifier_confidence"))},
        {"Метрика": "Источник вердикта", "Значение": _source_ru(result.get("verdict_source"))},
        {"Метрика": "Время обработки", "Значение": f"{result.get('processing_time_sec', '—')} сек"},
        {"Метрика": "Согласованность", "Значение": result.get("consistency_check", "")},
    ]


def _show_table(rows: list[dict[str, Any]]) -> None:
    if pd is not None:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:  # pragma: no cover
        st.table(rows)


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    verdict = result.get("verdict", "")
    return {
        "Файл": result.get("image_id", ""),
        "Сорт": C.CLASS_RU.get(verdict, verdict),
        "Тальк, %": result.get("talc_pct", ""),
        "Уверенность": _fmt_conf(result.get("classifier_confidence")),
        "Источник": _source_ru(result.get("verdict_source")),
        "Время, сек": result.get("processing_time_sec", ""),
        "Проверка": result.get("consistency_check", ""),
    }


def _download_pdf(result: dict[str, Any], original: np.ndarray, overlay: np.ndarray | None, key: str) -> None:
    try:
        pdf_path = report.build_pdf(
            result,
            original_png=_array_to_png_bytes(original),
            talc_overlay_png=_array_to_png_bytes(overlay),
        )
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            "Скачать PDF-отчёт",
            data=pdf_bytes,
            file_name=_safe_filename(str(result.get("image_id", "report")), ".pdf"),
            mime="application/pdf",
            key=f"pdf_{key}",
            width="stretch",
        )
    except Exception as exc:
        st.caption(f"PDF сейчас недоступен: {exc}")


def _review_payload(result: dict[str, Any], status: str, corrected: str | None, comment: str) -> dict[str, Any]:
    return {
        "image_id": result.get("image_id", ""),
        "model_verdict": result.get("verdict", ""),
        "expert_status": status,
        "expert_verdict": corrected,
        "expert_comment": comment,
        "talc_pct": result.get("talc_pct", ""),
        "consistency_check": result.get("consistency_check", ""),
    }


def _render_result(file_name: str, data: bytes, result: dict[str, Any], index: int) -> None:
    key = f"{index}_{_safe_filename(file_name, '')}"
    verdict = str(result.get("verdict", C.CLASS_ORDINARY))
    meta = _verdict_meta(verdict)
    tone, consistency = _consistency_badge(result)

    original = _decode_uploaded(data)
    overlay = _decode_png_b64(result.get("talc_mask_png_b64"))
    if overlay is None:
        overlay = original

    st.divider()
    st.subheader(file_name)
    st.markdown(
        f"""
        <div class="verdict-panel" style="border-left-color:{meta['color']}; background:{meta['soft']}">
            <div class="verdict-label">Сорт руды</div>
            <div class="verdict-title" style="color:{meta['color']}">{meta['title']}</div>
            <div class="verdict-text">{meta['description']}</div>
            <div style="margin-top:10px">
                <span class="status-badge {tone}">{consistency}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_metrics = st.columns(4)
    top_metrics[0].metric("Доля талька", _fmt_pct(result.get("talc_pct")))
    top_metrics[1].metric("Уверенность", _fmt_conf(result.get("classifier_confidence")))
    top_metrics[2].metric("Источник", _source_ru(result.get("verdict_source")))
    top_metrics[3].metric("Время", f"{result.get('processing_time_sec', '—')} сек")

    col_visual, col_details = st.columns([1.5, 1], gap="large")
    with col_visual:
        st.markdown("**Маска талька**")
        tab_mask, tab_original, tab_compare = st.tabs(["Маска", "Оригинал", "Сравнение"])
        with tab_mask:
            _interactive_viewer(overlay, key=f"{key}_overlay", height=540)
        with tab_original:
            _interactive_viewer(original, key=f"{key}_original", height=540)
        with tab_compare:
            left, right = st.columns(2)
            with left:
                st.caption("Оригинал")
                _interactive_viewer(original, key=f"{key}_compare_original", height=430)
            with right:
                st.caption("Маска талька")
                _interactive_viewer(overlay, key=f"{key}_compare_overlay", height=430)

    with col_details:
        st.markdown("**Таблица метрик**")
        _show_table(_metric_rows(result))

        st.markdown("**Экспертная проверка**")
        status = st.radio(
            "Решение геолога",
            ["Подтвердить вердикт", "Исправить сорт"],
            horizontal=True,
            key=f"review_status_{key}",
        )
        corrected = None
        if status == "Исправить сорт":
            corrected = st.selectbox(
                "Правильный сорт",
                options=C.CLASSES,
                format_func=lambda x: C.CLASS_RU.get(x, x),
                key=f"corrected_{key}",
            )
        comment = st.text_area("Комментарий", height=88, key=f"comment_{key}")
        review = _review_payload(result, status, corrected, comment)

        st.markdown('<div class="download-row">', unsafe_allow_html=True)
        _download_pdf(result, original, overlay, key)
        st.download_button(
            "Скачать JSON результата",
            data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=_safe_filename(str(result.get("image_id", "result")), ".json"),
            mime="application/json",
            key=f"json_{key}",
            width="stretch",
        )
        st.download_button(
            "Скачать экспертную правку",
            data=json.dumps(review, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=_safe_filename(str(result.get("image_id", "review")), "_review.json"),
            mime="application/json",
            key=f"review_{key}",
            width="stretch",
        )
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _inject_css()

    with st.sidebar:
        st.header("P4")
        st.caption("Локальный интерфейс сдачи: UI, отчёты, экспертная проверка.")
        st.link_button("Открыть текущий деплой", DEPLOY_URL, width="stretch")
        st.divider()
        st.write("Классы")
        for cls, meta in CLASS_META.items():
            st.markdown(
                f"<span class='status-badge' style='border-color:{meta['color']}; color:{meta['color']}'>{meta['title']}</span>",
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption(f"Порог согласованности талька: {C.TALC_VERDICT_THRESHOLD_PCT:.0f}%")

    st.title(APP_TITLE)
    st.markdown(
        "<div class='app-kicker'>Загрузите фото аншлифа: интерфейс покажет сорт, маску талька, метрики и отчёт для сдачи.</div>",
        unsafe_allow_html=True,
    )

    files = st.file_uploader(
        "Снимки аншлифа / панорамы",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
    )

    if not files:
        st.info("Ожидаю изображение для анализа.")
        return

    results: list[dict[str, Any]] = []
    for idx, file in enumerate(files):
        data = file.getvalue()
        if not data:
            st.warning(f"{file.name}: пустой файл пропущен.")
            continue
        try:
            with st.spinner(f"Анализирую {file.name}..."):
                result = _cached_analyze(data, file.name)
            results.append(result)
            _render_result(file.name, data, result, idx)
        except Exception as exc:
            st.error(f"Не удалось обработать «{file.name}»: {exc}")

    if not results:
        return

    st.divider()
    st.subheader("Сводка по загрузке")
    _show_table([_summary_row(r) for r in results])
    st.download_button(
        "Скачать CSV по всем изображениям",
        data=report.results_to_csv(results).encode("utf-8-sig"),
        file_name="analysis.csv",
        mime="text/csv",
        width="stretch",
    )


if __name__ == "__main__":
    main()
