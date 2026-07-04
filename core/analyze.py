# ============================================================
# ВЛАДЕЛЕЦ: P4 (интеграция контракта) — согласовывать с P1/P2 при смене полей
# TODO(P4): при появлении карты уверенности — добавить поле confidence_map_b64;
#           не менять имена полей без апдейта api/main.py и app/main.py.
# ============================================================
"""
analyze(image) -> dict — единый контракт, вокруг которого построены API и UI.

Контракт (по уточнениям жюри):
{
  "image_id": str,
  "verdict": "ordinary" | "hard_to_process" | "talc",   # ТОЛЬКО от классификатора
  "talc_pct": float,                                     # от тальк-сегментации
  "talc_mask_png_b64": str,                              # синяя маска талька поверх снимка
  "consistency_check": str,                              # RU: согласуется ли вердикт с %талька
  "classifier_confidence": float | null,
  "processing_time_sec": float
}

Деградация:
  - нет cv2/сегментации или ANALYZE_STUB=1 -> заглушка (фейковые числа, пустая маска);
  - нет весов классификатора -> fallback вердикта: talc_pct>10% => talc, иначе ordinary.
Разделение ordinary/hard_to_process делает ТОЛЬКО классификатор (никакой морфологии).
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

import config as C

try:
    import cv2
    import numpy as np
    from core import segment
    _HAS_SEG = True
except Exception as _e:  # pragma: no cover
    _HAS_SEG = False
    _SEG_ERR = _e

logging.basicConfig(
    filename=str(C.ANALYZE_LOG), level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("analyze")

ImageInput = Union[str, Path, bytes, "np.ndarray"]


# --------------------------------------------------------------------------- #
# Вердикт (только классификатор; fallback без весов)
# --------------------------------------------------------------------------- #
def decide_verdict(talc_pct: float, classifier_pred: Optional[dict]) -> tuple[str, Optional[float]]:
    """Возвращает (verdict, confidence). Вердикт берём у классификатора as-is
    (согласованность проверяем отдельно, не переопределяем). Без весов — грубый
    fallback: талька много => talc, иначе ordinary-заглушка."""
    if classifier_pred is not None:
        probs = classifier_pred.get("probs") or {}
        conf = max(probs.values()) if probs else None
        return classifier_pred["verdict"], conf
    if talc_pct > C.TALC_VERDICT_THRESHOLD_PCT:
        return C.CLASS_TALC, None
    return C.CLASS_ORDINARY, None


def consistency_check(verdict: str, talc_pct: float) -> str:
    """RU-строка: сходится ли вердикт классификатора с долей талька."""
    thr = C.TALC_VERDICT_THRESHOLD_PCT
    v_ru = C.CLASS_RU.get(verdict, verdict)
    if verdict == C.CLASS_TALC:
        if talc_pct >= thr:
            return f"вердикт talc согласуется: талька {talc_pct}%"
        return f"вердикт talc, но талька {talc_pct}% (<{thr:.0f}%) — проверить"
    if talc_pct >= thr:
        return f"вердикт {v_ru}, но талька {talc_pct}% (>{thr:.0f}%) — возможно talc, проверить"
    return f"вердикт {v_ru} согласуется: талька {talc_pct}%"


# --------------------------------------------------------------------------- #
# Кодирование
# --------------------------------------------------------------------------- #
def _png_b64(bgr: "np.ndarray") -> str:
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _pil_png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _source_to_pil_rgb(source: ImageInput):
    from PIL import Image

    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(source))).convert("RGB")
    if hasattr(source, "shape"):
        arr = source
        if getattr(arr, "ndim", 0) == 2:
            return Image.fromarray(arr).convert("RGB")
        if getattr(arr, "ndim", 0) == 3:
            return Image.fromarray(arr[..., :3]).convert("RGB")
    raise ValueError("unsupported image source for demo overlay")


def _stub_overlay_b64(source: ImageInput) -> str:
    """Visible no-cv2 demo overlay for local videos."""
    try:
        from PIL import Image, ImageChops, ImageFilter

        img = _source_to_pil_rgb(source)
        max_side = 2200
        if max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                             Image.Resampling.LANCZOS)

        gray = img.convert("L").filter(ImageFilter.GaussianBlur(radius=max(2, min(img.size) // 180)))
        values = list(gray.getdata())
        if not values:
            return ""

        idx = max(0, min(len(values) - 1, int(len(values) * 0.075)))
        threshold = sorted(values)[idx]
        mask = gray.point(lambda p: 255 if p <= threshold else 0, "L")
        mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(11))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1))

        alpha = mask.point(lambda p: int(p * 0.58))
        blue = Image.new("RGB", img.size, (0, 56, 220))
        out = Image.composite(blue, img, alpha)

        edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(7)),
                                   mask.filter(ImageFilter.MinFilter(7)))
        edge_alpha = edge.point(lambda p: 230 if p > 0 else 0)
        outline = Image.new("RGB", img.size, (0, 34, 190))
        out = Image.composite(outline, out, edge_alpha)
        return _pil_png_b64(out)
    except Exception as e:
        log.info("stub overlay unavailable for demo (%s)", type(e).__name__)
        return ""


def _image_id(source: ImageInput, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if isinstance(source, (str, Path)):
        return Path(source).stem
    if isinstance(source, (bytes, bytearray)):
        return "img_" + hashlib.md5(bytes(source)).hexdigest()[:10]
    return "img_" + time.strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- #
# Заглушка
# --------------------------------------------------------------------------- #
def _stub_result(image_id: str, t0: float, reason: str, image: ImageInput) -> dict:
    log.info("stub result for %s (%s)", image_id, reason)
    talc_pct = 4.0
    verdict = C.CLASS_ORDINARY
    return {
        "image_id": image_id,
        "verdict": verdict,
        "talc_pct": talc_pct,
        "talc_mask_png_b64": _stub_overlay_b64(image),
        "consistency_check": consistency_check(verdict, talc_pct),
        "classifier_confidence": None,
        "processing_time_sec": round(time.time() - t0, 3),
        "verdict_source": f"stub:{reason}",
    }


# --------------------------------------------------------------------------- #
# Контракт
# --------------------------------------------------------------------------- #
def analyze(image: ImageInput, image_id: Optional[str] = None,
            use_classifier: bool = True) -> dict:
    """Анализ одного снимка. Форма ответа — см. докстрок модуля."""
    t0 = time.time()
    iid = _image_id(image, image_id)

    if not _HAS_SEG or os.environ.get("ANALYZE_STUB") == "1":
        reason = "ANALYZE_STUB" if _HAS_SEG else f"no_segmentation({type(_SEG_ERR).__name__})"
        return _stub_result(iid, t0, reason, image)

    try:
        seg = segment.analyze_image(image)
    except Exception as e:  # не роняем API на кривом снимке
        log.exception("segmentation failed for %s: %s", iid, e)
        return _stub_result(iid, t0, f"seg_error:{type(e).__name__}", image)

    talc_pct = seg["talc_pct"]

    # Трек 1: классификатор (опционально; torch импортим лениво)
    classifier_pred = None
    if use_classifier:
        try:
            from core import classifier
            classifier_pred = classifier.predict(seg["bgr"])
        except Exception as e:
            log.info("classifier unavailable (%s) — fallback verdict", type(e).__name__)

    verdict, conf = decide_verdict(talc_pct, classifier_pred)
    source = "classifier" if classifier_pred is not None else (
        "fallback:talc>thr" if talc_pct > C.TALC_VERDICT_THRESHOLD_PCT else "fallback:default")

    result = {
        "image_id": iid,
        "verdict": verdict,
        "talc_pct": talc_pct,
        "talc_mask_png_b64": _png_b64(seg["overlay"]),
        "consistency_check": consistency_check(verdict, talc_pct),
        "classifier_confidence": conf,
        "processing_time_sec": round(time.time() - t0, 3),
        "verdict_source": source,
    }
    log.info("analyzed %s verdict=%s (%s) talc=%.2f%% in %.2fs",
             iid, verdict, source, talc_pct, result["processing_time_sec"])
    return result


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1:
        r = analyze(sys.argv[1])
        r = {k: (f"<{len(v)} b64 chars>" if k == "talc_mask_png_b64" and v else v)
             for k, v in r.items()}
        print(json.dumps(r, ensure_ascii=False, indent=2))
