# ============================================================
# ВЛАДЕЛЕЦ: P4 (отчёты CSV/PDF)
# TODO(P4): при демо — проверить, что Cyrillic-шрифт есть на сервере
#           (иначе PDF-текст «квадратами»); при желании добавить логотип/шапку.
# ============================================================
"""
Отчёты: RU текст-вердикт, CSV, PDF. Только тальк + сорт (по уточнениям жюри).

PDF на reportlab. Для кириллицы нужен Unicode-TTF — пробуем зарегистрировать
DejaVuSans (есть на сервере Ubuntu); иначе Helvetica (кириллица может не отрисоваться).
"""
from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import config as C

log = logging.getLogger("report")

_CSV_FIELDS = ["image_id", "verdict", "talc_pct", "classifier_confidence",
               "consistency_check", "processing_time_sec"]

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


# --------------------------------------------------------------------------- #
# Текст-вердикт (RU)
# --------------------------------------------------------------------------- #
def verdict_text(result: dict) -> str:
    """Краткое заключение, напр.:
    'Руда классифицирована как оталькованная: тальк — 14.0%. вердикт talc согласуется: талька 14.0%'."""
    verdict = result.get("verdict", C.CLASS_ORDINARY)
    ru = C.CLASS_RU.get(verdict, verdict)
    return (f"Руда классифицирована как {ru}: тальк — {result.get('talc_pct', 0.0)}%. "
            f"{result.get('consistency_check', '')}").strip()


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def _row(result: dict) -> dict:
    return {
        "image_id": result.get("image_id", ""),
        "verdict": result.get("verdict", ""),
        "talc_pct": result.get("talc_pct", ""),
        "classifier_confidence": result.get("classifier_confidence", ""),
        "consistency_check": result.get("consistency_check", ""),
        "processing_time_sec": result.get("processing_time_sec", ""),
    }


def results_to_csv(results: Iterable[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CSV_FIELDS)
    w.writeheader()
    for r in results:
        w.writerow(_row(r))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _register_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Cyr", path))
                return "Cyr"
            except Exception:
                continue
    log.warning("Cyrillic TTF не найден — PDF на Helvetica (кириллица может не отрисоваться).")
    return "Helvetica"


def _safe_report_stem(image_id: str) -> str:
    """Keep uploaded filenames from becoming nested report paths."""
    stem = Path(str(image_id) or "report").name
    stem = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", stem).strip("._")
    return stem or "report"


def build_pdf(result: dict, out_path: Optional[Path] = None,
              original_png: Optional[bytes] = None,
              talc_overlay_png: Optional[bytes] = None) -> Path:
    """Одностраничный PDF: вердикт сорта, тальк %, согласованность, снимок + маска талька."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    image_id = result.get("image_id", "report")
    out_path = Path(out_path) if out_path else (C.REPORTS_DIR / f"{_safe_report_stem(image_id)}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    font = _register_font()
    h1 = ParagraphStyle("h1", fontName=font, fontSize=16, leading=20, spaceAfter=8)
    body = ParagraphStyle("body", fontName=font, fontSize=11, leading=15)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    verdict = result.get("verdict", C.CLASS_ORDINARY)
    ru = C.CLASS_RU.get(verdict, verdict)
    conf = result.get("classifier_confidence")
    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
    story = [
        Paragraph("Отчёт анализа аншлифа", h1),
        Paragraph(f"ID образца: {image_id}", body),
        Spacer(1, 6),
        Paragraph(f"Сорт руды (вердикт): <b>{ru}</b> (уверенность: {conf_str})", body),
        Paragraph(verdict_text(result), body),
        Spacer(1, 10),
    ]

    data = [
        ["Метрика", "Значение"],
        ["Сорт руды", ru],
        ["Доля талька, %", result.get("talc_pct", 0.0)],
        ["Согласованность", result.get("consistency_check", "")],
    ]
    table = Table(data, colWidths=[6 * cm, 9 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f2f6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [table, Spacer(1, 14)]

    for caption, png in (("Исходное изображение", original_png),
                         ("Маска талька (синий)", talc_overlay_png)):
        if png:
            story.append(Paragraph(caption, body))
            story.append(RLImage(io.BytesIO(png), width=15 * cm, height=11 * cm, kind="proportional"))
            story.append(Spacer(1, 8))

    doc.build(story)
    return out_path
