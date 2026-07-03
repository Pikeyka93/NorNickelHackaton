"""
Central configuration: paths, class names, thresholds.

Everything tunable lives here so the 3 tracks (classifier / segmentation / API-UI)
share one source of truth. Values that need calibration on the real data are marked
with TODO and carry a sane default so the code runs out of the box.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Repo root -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val) if val else default


# --- Data ------------------------------------------------------------------
# Dataset lives on the server (cyrillic + spaces). NEVER commit it (see .gitignore).
# Override with:  export DATA_DIR="/some/path"
DATA_DIR = _env_path(
    "DATA_DIR",
    Path.home() / "dataset" / "Задача 3. Скажи мне, кто твой шлиф",
)

# Subfolders inside DATA_DIR that hold the per-class images.
PART1_DIR = "Фото руд по сортам. ч1"
PART2_DIR = "Фото руд по сортам. ч2"
PANORAMA_DIR = "Панорамы"

# --- Artifacts (kept out of git) -------------------------------------------
WEIGHTS_DIR = _env_path("WEIGHTS_DIR", ROOT / "weights")
CLASSIFIER_WEIGHTS = _env_path("CLASSIFIER_WEIGHTS", WEIGHTS_DIR / "classifier.pt")
REPORTS_DIR = _env_path("REPORTS_DIR", ROOT / "reports")
LOG_DIR = _env_path("LOG_DIR", ROOT / "logs")

for _d in (WEIGHTS_DIR, REPORTS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Canonical classes -----------------------------------------------------
CLASS_ORDINARY = "ordinary"          # рядовая руда
CLASS_HARD = "hard_to_process"       # труднообогатимая / тонкая
CLASS_TALC = "talc"                  # оталькованная
CLASSES = [CLASS_ORDINARY, CLASS_HARD, CLASS_TALC]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Human-readable (RU) labels for reports / UI.
CLASS_RU = {
    CLASS_ORDINARY: "рядовая",
    CLASS_HARD: "труднообогатимая",
    CLASS_TALC: "оталькованная",
}

# --- Verdict logic ---------------------------------------------------------
# Expert rule: talc > 10% => talc; else ordinary vs fine by which dominates.
TALC_VERDICT_THRESHOLD_PCT = 10.0

# --- Classifier ------------------------------------------------------------
CLASSIFIER_BACKBONE = os.environ.get("CLASSIFIER_BACKBONE", "efficientnet_b0")  # or "resnet50"
IMG_SIZE = 320                       # input resolution for the classifier
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
LR = float(os.environ.get("LR", "3e-4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
VAL_FRACTION = 0.2
RANDOM_SEED = 42
# ImageNet normalisation (pretrained backbones).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Segmentation (classical CV) -------------------------------------------
# Tiling for panoramas: process in overlapping tiles, never load-and-process 10k^2 at once.
TILE_SIZE = 1024
TILE_OVERLAP = 128
# Guard against pathological loads; above this longest side we downscale before masking
# (percentages are scale-invariant). TODO: swap for true lazy tiling (pyvips/tifffile).
MAX_PROCESS_SIDE = 8000

# Sulfide detection: bright phase. Otsu offset lets us bias the auto threshold.
# TODO: calibrate offset on labelled slides.
SULFIDE_OTSU_OFFSET = 0.0            # added to Otsu threshold (0-255 scale)
SULFIDE_MIN_BLOB_AREA = 25          # px; drop specks below this after thresholding

# Ordinary vs fine intergrowth (per sulfide blob morphology).
# Big / compact / high-solidity -> ordinary (green); small / ragged -> fine (red).
# TODO: calibrate these cut-offs against expert examples.
INTERGROWTH_SOLIDITY_CUT = 0.85     # >= => tends ordinary
INTERGROWTH_AREA_CUT = 1500         # px; >= => tends ordinary
INTERGROWTH_COMPACTNESS_CUT = 30.0  # perimeter^2/area; < => compact => ordinary

# Talc: dark, scattered, low-texture phase in the non-sulfide matrix.
# TODO: calibrate against the expert BLUE outlines (target talc error <= +-3%).
TALC_DARK_PERCENTILE = 25           # pixels darker than this local percentile are talc candidates
TALC_LOCAL_WINDOW = 51              # px; window for local adaptive darkness
TALC_MAX_TEXTURE = 0.06             # normalised local variance ceiling (smooth => talc)
TALC_MIN_BLOB_AREA = 200            # px; drop tiny talc specks

# Blue expert-outline extraction (calibration only) in HSV.
BLUE_HSV_LOWER = (90, 60, 40)
BLUE_HSV_UPPER = (140, 255, 255)

# --- Colours for the phase mask (BGR for OpenCV) ---------------------------
COLOR_ORDINARY = (0, 200, 0)     # green
COLOR_FINE = (0, 0, 220)         # red
COLOR_TALC = (220, 0, 0)         # blue
MASK_ALPHA = 0.45                # overlay opacity

# --- API / logging ---------------------------------------------------------
ANALYZE_LOG = LOG_DIR / "analyze.log"
