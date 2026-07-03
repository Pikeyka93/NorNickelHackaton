"""
Classical-CV segmentation of aншлиф phases -> color mask + area percentages.

Pipeline (per the brief):
  1. per-image (per-tile) illumination normalisation  -- MUST come first
  2. sulfide detection: Otsu on the *normalised* brightness
  3. ordinary vs fine intergrowth: morphology of sulfide blobs (solidity/compactness)
  4. talc: locally-dark + low-texture regions in the non-sulfide matrix
  5. compose green/red/blue mask, compute % of total slide area
Panoramas are processed in overlapping TILES so we never run heavy ops on 10k^2 at once.

`extract_blue_annotations` pulls the expert's blue outlines — the only pixel-level
ground truth we have — for calibrating the talc detector (see notebooks/).

All thresholds live in config.py with TODO calibration notes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

import config as C

# skimage is optional: we use LBP texture if present, else a cv2 local-variance fallback.
try:
    from skimage.feature import local_binary_pattern  # noqa: F401
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False

# Allow very large panoramas through PIL if we ever route via it.
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except Exception:  # pragma: no cover
    Image = None

ImageInput = Union[str, Path, bytes, np.ndarray]
_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_image_bgr(source: ImageInput, max_side: int = C.MAX_PROCESS_SIDE) -> np.ndarray:
    """Load any supported input into a BGR uint8 array, downscaling if a side exceeds
    `max_side` (percentages are scale-invariant, so this is safe and bounds memory).
    TODO: for true multi-GB TIFFs, replace with lazy tiling via pyvips/tifffile."""
    if isinstance(source, np.ndarray):
        img = source
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes.")
    else:  # path
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- #
# Step 1: illumination normalisation (per tile)
# --------------------------------------------------------------------------- #
def normalize_illumination(bgr: np.ndarray) -> np.ndarray:
    """Gray-world white balance + large-scale flat-field division.

    Kills the yellow/dark/bright colour casts so downstream thresholds see comparable
    brightness across wildly different captures. A FIXED threshold on the raw image
    would not survive this variation — hence normalise first, always."""
    f = bgr.astype(np.float32)

    # 1) gray-world white balance: pull each channel to a common mean.
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    f *= (means.mean() / means)

    # 2) flat-field: divide luminance by a heavily-blurred estimate of the background.
    gray = f.mean(axis=2)
    k = max(31, (min(bgr.shape[:2]) // 8) | 1)  # odd kernel ~1/8 of the shorter side
    bg = cv2.GaussianBlur(gray, (k, k), 0) + 1e-6
    gain = (gray.mean() / bg)[..., None]
    f *= gain

    return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Step 2: sulfides (bright phase)
# --------------------------------------------------------------------------- #
def segment_sulfides(norm_bgr: np.ndarray) -> np.ndarray:
    """Binary mask (0/255) of bright sulfide grains via Otsu on the normalised gray."""
    gray = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu + C.SULFIDE_OTSU_OFFSET, 0, 255))
    mask = (gray >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3)
    return _drop_small(mask, C.SULFIDE_MIN_BLOB_AREA)


# --------------------------------------------------------------------------- #
# Step 4: talc (dark, smooth, in the matrix)
# --------------------------------------------------------------------------- #
def _local_variance(gray_f: np.ndarray, win: int) -> np.ndarray:
    """Normalised (0..1) local variance via box filters."""
    mean = cv2.boxFilter(gray_f, -1, (win, win))
    mean_sq = cv2.boxFilter(gray_f * gray_f, -1, (win, win))
    var = np.clip(mean_sq - mean * mean, 0, None)
    return var / (255.0 ** 2)


def detect_talc(norm_bgr: np.ndarray, sulfide_mask: np.ndarray) -> np.ndarray:
    """Talc = matrix pixels that are (a) dark globally, (b) darker than local surroundings,
    (c) low-texture. Calibrate the thresholds against the blue expert outlines."""
    gray = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    matrix = sulfide_mask == 0
    if matrix.sum() < 10:
        return np.zeros(gray.shape, np.uint8)

    # (a) global darkness within the matrix
    dark_thr = np.percentile(gray[matrix], C.TALC_DARK_PERCENTILE)
    dark_global = gray < dark_thr

    # (b) locally darker than surroundings
    win = C.TALC_LOCAL_WINDOW | 1
    local_mean = cv2.boxFilter(gray, -1, (win, win))
    dark_local = gray < (local_mean - 5.0)

    # (c) smooth (low local variance)
    smooth = _local_variance(gray, win) < C.TALC_MAX_TEXTURE

    talc = (matrix & dark_global & dark_local & smooth).astype(np.uint8) * 255
    talc = cv2.morphologyEx(talc, cv2.MORPH_OPEN, _K5)
    talc = cv2.morphologyEx(talc, cv2.MORPH_CLOSE, _K5)
    return _drop_small(talc, C.TALC_MIN_BLOB_AREA)


# --------------------------------------------------------------------------- #
# Step 3: ordinary vs fine intergrowths (blob morphology, global)
# --------------------------------------------------------------------------- #
def classify_intergrowths(sulfide_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the sulfide mask into ordinary (green) and fine (red) by per-blob shape.

    Big / compact / high-solidity blobs -> ordinary; small / ragged / low-solidity -> fine.
    A blob is ordinary if it wins >=2 of the 3 morphology votes (config cut-offs)."""
    ordinary = np.zeros(sulfide_mask.shape, np.uint8)
    fine = np.zeros(sulfide_mask.shape, np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(sulfide_mask, 8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < C.SULFIDE_MIN_BLOB_AREA:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / (hull_area + 1e-6)
        compactness = (peri * peri) / (area + 1e-6)  # 4pi for a disk; grows when ragged

        votes = 0
        votes += solidity >= C.INTERGROWTH_SOLIDITY_CUT
        votes += area >= C.INTERGROWTH_AREA_CUT
        votes += compactness < C.INTERGROWTH_COMPACTNESS_CUT
        (ordinary if votes >= 2 else fine)[comp > 0] = 255

    return ordinary, fine


# --------------------------------------------------------------------------- #
# Blue expert outlines (calibration only)
# --------------------------------------------------------------------------- #
def extract_blue_annotations(bgr: np.ndarray, fill: bool = True) -> np.ndarray:
    """Mask of the expert's hand-drawn BLUE talc outlines. If `fill`, flood the interior
    so the mask is the annotated *region*, not just the line. Use as talc ground truth."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    line = cv2.inRange(hsv, np.array(C.BLUE_HSV_LOWER), np.array(C.BLUE_HSV_UPPER))
    if not fill:
        return line
    closed = cv2.morphologyEx(line, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros(bgr.shape[:2], np.uint8)
    cv2.drawContours(region, cnts, -1, 255, thickness=cv2.FILLED)
    return region


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #
def iter_tiles(h: int, w: int, tile: int, overlap: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield (y0, y1, x0, x1) overlapping tiles covering an h*w image."""
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - overlap), step))
    xs = list(range(0, max(1, w - overlap), step))
    for y0 in ys:
        y1 = min(h, y0 + tile)
        for x0 in xs:
            x1 = min(w, x0 + tile)
            yield y0, y1, x0, x1


def segment_masks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-tile normalise + sulfide + talc, stitched into full-res masks.
    Local ops (normalise/threshold) run per tile; global blob morphology runs afterwards."""
    h, w = bgr.shape[:2]
    sulf = np.zeros((h, w), np.uint8)
    talc = np.zeros((h, w), np.uint8)

    single = (h <= C.TILE_SIZE and w <= C.TILE_SIZE)
    tiles = [(0, h, 0, w)] if single else iter_tiles(h, w, C.TILE_SIZE, C.TILE_OVERLAP)
    for (y0, y1, x0, x1) in tiles:
        norm = normalize_illumination(bgr[y0:y1, x0:x1])
        s = segment_sulfides(norm)
        t = detect_talc(norm, s)
        # union in overlap regions
        sulf[y0:y1, x0:x1] = np.maximum(sulf[y0:y1, x0:x1], s)
        talc[y0:y1, x0:x1] = np.maximum(talc[y0:y1, x0:x1], t)

    talc[sulf > 0] = 0  # talc lives in the non-sulfide matrix
    return sulf, talc


# --------------------------------------------------------------------------- #
# Compose + metrics
# --------------------------------------------------------------------------- #
def build_color_mask(shape, ordinary, fine, talc) -> np.ndarray:
    """BGR mask: green=ordinary, red=fine, blue=talc, black elsewhere."""
    mask = np.zeros((shape[0], shape[1], 3), np.uint8)
    mask[talc > 0] = C.COLOR_TALC
    mask[fine > 0] = C.COLOR_FINE
    mask[ordinary > 0] = C.COLOR_ORDINARY
    return mask


def overlay_mask(bgr: np.ndarray, color_mask: np.ndarray, alpha: float = C.MASK_ALPHA) -> np.ndarray:
    out = bgr.copy()
    nz = color_mask.any(axis=2)
    out[nz] = cv2.addWeighted(bgr, 1 - alpha, color_mask, alpha, 0)[nz]
    return out


def compute_metrics(ordinary, fine, talc) -> dict:
    """Percentages of TOTAL slide area (all pixels). ordinary+fine ~= sulfide."""
    total = ordinary.shape[0] * ordinary.shape[1]
    o = int((ordinary > 0).sum())
    f = int((fine > 0).sum())
    t = int((talc > 0).sum())
    pct = lambda n: round(100.0 * n / total, 2)
    return {
        "sulfide_area_pct": pct(o + f),
        "ordinary_pct": pct(o),
        "fine_pct": pct(f),
        "talc_pct": pct(t),
    }


def analyze_image(source: ImageInput) -> dict:
    """Full segmentation. Returns metrics + BGR masks/overlay (kept as arrays; the
    analyze() contract base64-encodes them)."""
    bgr = read_image_bgr(source)
    sulf, talc = segment_masks(bgr)
    ordinary, fine = classify_intergrowths(sulf)
    color_mask = build_color_mask(bgr.shape, ordinary, fine, talc)
    return {
        "bgr": bgr,
        "metrics": compute_metrics(ordinary, fine, talc),
        "color_mask": color_mask,
        "overlay": overlay_mask(bgr, color_mask),
        "masks": {"sulfide": sulf, "ordinary": ordinary, "fine": fine, "talc": talc},
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Zero out connected components smaller than min_area."""
    if min_area <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = analyze_image(sys.argv[1])
        print("metrics:", res["metrics"])
