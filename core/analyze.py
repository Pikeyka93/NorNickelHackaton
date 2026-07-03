"""
analyze(image) -> dict  — the single contract the API and UI are built around.

Design:
  * Verdict comes from the CLASSIFIER (track 1); mask + percentages from SEGMENTATION
    (track 2). They are reconciled so the returned verdict never contradicts the numbers
    (talc>threshold always wins, per the expert rule).
  * Degrades gracefully:
      - no cv2/segmentation available OR ANALYZE_STUB=1  -> stub numbers + empty mask
        (lets the API/UI team work before the CV/ML is wired in).
      - no trained classifier weights                    -> rule-based verdict from the
        segmentation percentages.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

import config as C

# Segmentation is the heavy import; guard it so the stub path works without cv2.
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
# Verdict reconciliation (expert logic)
# --------------------------------------------------------------------------- #
def decide_verdict(metrics: dict, classifier_pred: Optional[dict]) -> tuple[str, str]:
    """Return (verdict, source). Rule order:
      1. talc% > threshold                         -> talc
      2. else classifier verdict (if it disagrees with numbers on talc, defer to numbers)
      3. else ordinary vs fine dominance from the segmentation
    """
    if metrics["talc_pct"] > C.TALC_VERDICT_THRESHOLD_PCT:
        return C.CLASS_TALC, "talc_over_threshold"

    dominance = C.CLASS_ORDINARY if metrics["ordinary_pct"] >= metrics["fine_pct"] else C.CLASS_HARD

    if classifier_pred is not None:
        v = classifier_pred["verdict"]
        # classifier says talc but segmentation talc is below threshold -> trust the numbers
        if v == C.CLASS_TALC:
            return dominance, "classifier_talc_low_defer_seg"
        return v, "classifier"

    return dominance, "rule_based"


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
def _png_b64(bgr_or_bgra: "np.ndarray") -> str:
    ok, buf = cv2.imencode(".png", bgr_or_bgra)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _mask_to_rgba_b64(color_mask_bgr: "np.ndarray") -> str:
    """Color mask -> transparent-background RGBA PNG (alpha=0 where no phase), so the UI
    can overlay it on the original at any opacity."""
    alpha = (color_mask_bgr.any(axis=2).astype("uint8")) * 255
    bgra = cv2.merge([color_mask_bgr[:, :, 0], color_mask_bgr[:, :, 1],
                      color_mask_bgr[:, :, 2], alpha])
    return _png_b64(bgra)


def _image_id(source: ImageInput, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if isinstance(source, (str, Path)):
        return Path(source).stem
    if isinstance(source, (bytes, bytearray)):
        return "img_" + hashlib.md5(bytes(source)).hexdigest()[:10]
    return "img_" + time.strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- #
# Stub (used when segmentation is unavailable or ANALYZE_STUB=1)
# --------------------------------------------------------------------------- #
def _stub_result(image_id: str, t0: float, reason: str) -> dict:
    log.info("stub result for %s (%s)", image_id, reason)
    return {
        "image_id": image_id,
        "verdict": C.CLASS_ORDINARY,
        "metrics": {"sulfide_area_pct": 12.5, "ordinary_pct": 9.0,
                    "fine_pct": 3.5, "talc_pct": 4.0},
        "mask_png_b64": "",
        "confidence_map_b64": "",
        "processing_time_sec": round(time.time() - t0, 3),
        "verdict_source": f"stub:{reason}",
        "classifier_probs": None,
    }


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
def analyze(image: ImageInput, image_id: Optional[str] = None,
            use_classifier: bool = True) -> dict:
    """Analyse one слайд. See module docstring for the contract shape."""
    t0 = time.time()
    iid = _image_id(image, image_id)

    if not _HAS_SEG or os.environ.get("ANALYZE_STUB") == "1":
        reason = "ANALYZE_STUB" if _HAS_SEG else f"no_segmentation({type(_SEG_ERR).__name__})"
        return _stub_result(iid, t0, reason)

    try:
        seg = segment.analyze_image(image)
    except Exception as e:  # never let the API 500 on a bad image — return a stub
        log.exception("segmentation failed for %s: %s", iid, e)
        return _stub_result(iid, t0, f"seg_error:{type(e).__name__}")

    metrics = seg["metrics"]

    # Track 1: classifier verdict (optional; lazy import so torch stays optional).
    classifier_pred = None
    if use_classifier:
        try:
            from core import classifier
            classifier_pred = classifier.predict(seg["bgr"])
        except Exception as e:
            log.info("classifier unavailable (%s) — using rule-based verdict", type(e).__name__)

    verdict, source = decide_verdict(metrics, classifier_pred)

    result = {
        "image_id": iid,
        "verdict": verdict,
        "metrics": metrics,
        "mask_png_b64": _mask_to_rgba_b64(seg["color_mask"]),
        "confidence_map_b64": "",   # TODO: expose classifier heatmap / seg certainty
        "processing_time_sec": round(time.time() - t0, 3),
        "verdict_source": source,
        "classifier_probs": classifier_pred["probs"] if classifier_pred else None,
    }
    log.info("analyzed %s verdict=%s (%s) metrics=%s in %.2fs",
             iid, verdict, source, metrics, result["processing_time_sec"])
    return result


def overlay_png_b64(image: ImageInput) -> str:
    """Convenience for the report/UI: full mask-over-image overlay as base64 PNG."""
    if not _HAS_SEG:
        return ""
    seg = segment.analyze_image(image)
    return _png_b64(seg["overlay"])


if __name__ == "__main__":
    import json
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if src:
        r = analyze(src)
        r_print = {k: (v if k not in ("mask_png_b64", "confidence_map_b64")
                       else f"<{len(v)} b64 chars>") for k, v in r.items()}
        print(json.dumps(r_print, ensure_ascii=False, indent=2))
