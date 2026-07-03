# ============================================================
# ВЛАДЕЛЕЦ: P3 (API + Docker + стабильность деплоя)
# TODO(P3):
#   - зафиксировать torch/torchvision под cu121 (requirements.txt / Dockerfile);
#   - следить, чтобы правки не роняли живой деплой (per-file try/except уже есть);
#   - проверить /report/pdf и /export/csv на реальных снимках.
# ============================================================
"""
FastAPI:  upload -> analyze() -> JSON (новый контракт: talc_pct + маска талька),
плюс batch, CSV, PDF, логирование.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import io
import logging
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import config as C
from core import analyze as analyze_mod
from core import report

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI(
    title="Скажи мне, кто твой шлиф — API",
    description="Сорт руды (классификатор) + сегментация талька по аншлифу.",
    version="0.2.0",
)


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Пустой файл: {file.filename}")
    return data


def _png_bytes(bgr, max_side: int = 1400) -> bytes:
    """Уменьшенный PNG BGR-массива (для вставки в отчёт)."""
    import cv2
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else b""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": C.CLASSES, "segmentation": analyze_mod._HAS_SEG}


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Анализ одного снимка -> JSON контракта (verdict, talc_pct, маска талька, ...)."""
    data = await _read(file)
    return JSONResponse(analyze_mod.analyze(data, image_id=file.filename))


@app.post("/analyze/batch")
async def analyze_batch(files: List[UploadFile] = File(...)) -> JSONResponse:
    """Пакетная обработка. Ошибка отдельного файла не валит весь batch."""
    results = []
    for f in files:
        try:
            results.append(analyze_mod.analyze(await _read(f), image_id=f.filename))
        except Exception as e:
            log.exception("batch item failed: %s", f.filename)
            results.append({"image_id": f.filename, "error": str(e)})
    return JSONResponse({"count": len(results), "results": results})


@app.post("/export/csv")
async def export_csv(files: List[UploadFile] = File(...)) -> StreamingResponse:
    """Batch -> CSV-выгрузка метрик (image_id, verdict, talc_pct, ...)."""
    results = [analyze_mod.analyze(await _read(f), image_id=f.filename) for f in files]
    csv_text = report.results_to_csv([r for r in results if "error" not in r])
    return StreamingResponse(
        io.BytesIO(csv_text.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=analysis.csv"},
    )


@app.post("/report/pdf")
async def report_pdf(file: UploadFile = File(...)) -> StreamingResponse:
    """Анализ одного снимка -> PDF-отчёт (сорт + тальк + маска)."""
    data = await _read(file)
    result = analyze_mod.analyze(data, image_id=file.filename)

    original_png = talc_overlay_png = None
    if analyze_mod._HAS_SEG:
        try:
            from core import segment
            seg = segment.analyze_image(data)
            original_png = _png_bytes(seg["bgr"])
            talc_overlay_png = _png_bytes(seg["overlay"])
        except Exception:
            log.exception("не удалось отрисовать картинки для отчёта")

    pdf_path = report.build_pdf(result, original_png=original_png,
                                talc_overlay_png=talc_overlay_png)
    return StreamingResponse(
        open(pdf_path, "rb"),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{result["image_id"]}.pdf"'},
    )
