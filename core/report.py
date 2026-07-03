"""
Report generation: RU text verdict, CSV export, PDF report.

PDF uses reportlab. Cyrillic needs a Unicode TTF — we try to register DejaVuSans from
common locations (present on the Ubuntu server); if none is found we fall back to
Helvetica and log a warning (PDF still generates, Cyrillic may render as boxes).
"""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Iterable, Optional

import config as C

log = logging.getLogger("report")

_CSV_FIELDS = ["image_id", "verdict", "sulfide_area_pct", "ordinary_pct",
               "fine_pct", "talc_pct", "verdict_source", "processing_time_sec"]

# Candidate Cyrillic-capable TTFs (server first, then mac).
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


# --------------------------------------------------------------------------- #
# Text verdict (RU)
# --------------------------------------------------------------------------- #
def verdict_text(result: dict) -> str:
    """One-line RU conclusion, e.g.:
    'Руда классифицирована как оталькованная: тальк — 14.0%, преобладание тонких срастаний — 62%.'"""
    m = result["metrics"]
    ru = C.CLASS_RU.get(result["verdict"], result["verdict"])
    total_sulf = max(m["sulfide_area_pct"], 1e-6)
    ord_share = round(100 * m["ordinary_pct"] / total_sulf)
    fine_share = round(100 * m["fine_pct"] / total_sulf)
    dominant = ("тонких срастаний", fine_share) if m["fine_pct"] > m["ordinary_pct"] \
        else ("обычных срастаний", ord_share)
    return (f"Руда классифицирована как {ru}: "
            f"тальк — {m['talc_pct']}%, "
            f"сульфиды — {m['sulfide_area_pct']}%, "
            f"преобладание {dominant[0]} — {dominant[1]}%.")


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def _row(result: dict) -> dict:
    m = result["metrics"]
    return {
        "image_id": result["image_id"],
        "verdict": result["verdict"],
        "sulfide_area_pct": m["sulfide_area_pct"],
        "ordinary_pct": m["ordinary_pct"],
        "fine_pct": m["fine_pct"],
        "talc_pct": m["talc_pct"],
        "verdict_source": result.get("verdict_source", ""),
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
    log.warning("No Cyrillic TTF found — PDF falls back to Helvetica (Cyrillic may not render).")
    return "Helvetica"


def build_pdf(result: dict, out_path: Optional[Path] = None,
              original_png: Optional[bytes] = None,
              overlay_png: Optional[bytes] = None) -> Path:
    """Render a one-page PDF: verdict, RU text, metrics table, original + mask overlay."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    out_path = Path(out_path) if out_path else (C.REPORTS_DIR / f"{result['image_id']}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    font = _register_font()
    h1 = ParagraphStyle("h1", fontName=font, fontSize=16, leading=20, spaceAfter=8)
    body = ParagraphStyle("body", fontName=font, fontSize=11, leading=15)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = [
        Paragraph("Отчёт анализа аншлифа", h1),
        Paragraph(f"ID образца: {result['image_id']}", body),
        Spacer(1, 6),
        Paragraph(verdict_text(result), body),
        Spacer(1, 10),
    ]

    m = result["metrics"]
    data = [
        ["Метрика", "Значение, %"],
        ["Доля сульфидов", m["sulfide_area_pct"]],
        ["Обычные срастания (зелёный)", m["ordinary_pct"]],
        ["Тонкие срастания (красный)", m["fine_pct"]],
        ["Тальк (синий)", m["talc_pct"]],
    ]
    table = Table(data, colWidths=[9 * cm, 5 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f2f6")]),
    ]))
    story += [table, Spacer(1, 14)]

    for caption, png in (("Исходное изображение", original_png), ("Маска фаз", overlay_png)):
        if png:
            story.append(Paragraph(caption, body))
            story.append(RLImage(io.BytesIO(png), width=15 * cm, height=11 * cm, kind="proportional"))
            story.append(Spacer(1, 8))

    doc.build(story)
    return out_path
