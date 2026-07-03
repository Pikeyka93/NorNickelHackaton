# ============================================================
# ВЛАДЕЛЕЦ: общий (менять согласованно — от config зависят все треки)
# TALC_*: откалиброваны scripts/calibrate_talc.py по 42 парам (оригинал + синяя
# обводка) из "Оталькованные руды/Области оталькования" — mean_abs_err_pct
# 4.92% (было 14.83% на дефолтах), цель брифа ±3%. Отчёт: /tmp/final_applied.json
# (локально, не в репозитории). Если данных на сервере больше/иначе — можно
# перекалибровать: python scripts/calibrate_talc.py --resume-from <тот отчёт> --apply
# TODO(P2): при желании вынести гиперпараметры классификатора в отдельный конфиг
# ============================================================
"""
Central configuration: paths, class names, thresholds.

Scope (по уточнениям жюри): сегментируем ТОЛЬКО тальк; сорт руды определяет
классификатор. Порогов срастаний здесь больше нет.
Значения, требующие калибровки на данных, помечены TODO и имеют рабочий дефолт.
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
# Датасет на сервере (кириллица + пробелы). НЕ коммитить. Override: export DATA_DIR=...
DATA_DIR = _env_path(
    "DATA_DIR",
    Path.home() / "dataset" / "Задача 3. Скажи мне, кто твой шлиф",
)

# Подпапки внутри DATA_DIR
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

CLASS_RU = {
    CLASS_ORDINARY: "рядовая",
    CLASS_HARD: "труднообогатимая",
    CLASS_TALC: "оталькованная",
}

# --- Verdict / consistency -------------------------------------------------
# Порог для эвристики согласованности и для fallback-вердикта без классификатора.
TALC_VERDICT_THRESHOLD_PCT = 10.0

# --- Classifier ------------------------------------------------------------
CLASSIFIER_BACKBONE = os.environ.get("CLASSIFIER_BACKBONE", "efficientnet_b0")  # или "resnet50"
IMG_SIZE = 320
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
LR = float(os.environ.get("LR", "3e-4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
VAL_FRACTION = 0.2
RANDOM_SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Talc segmentation (classical CV) --------------------------------------
# Тальк = рассеянная ТЁМНАЯ фаза в нерудной матрице. Детектор: локально-адаптивный
# порог «темнее локального фона» + низкая текстура. Фиксированный порог по сырому
# снимку не работает (снимки от почти чёрных до жёлтых) => сначала нормализация.
# TODO(P1): откалибровать по синим обводкам, цель talc_pct ±3%.
TALC_BRIGHT_EXCLUDE_PERCENTILE = 92  # ярче этого => рудная/светлая фаза, не тальк
TALC_DARK_PERCENTILE = 32            # перцентиль матрицы как уровень нерудного фона (50 = медиана)
TALC_ABS_MARGIN = 32.2               # на столько темнее фона по АБСОЛЮТУ (ловит интерьер больших зон)
TALC_LOCAL_WINDOW = 101              # px, окно оценки локального фона (нечётное)
TALC_LOCAL_MARGIN = 14.2              # на столько темнее ЛОКАЛЬНОГО фона (края/остаточный градиент)
TALC_MAX_TEXTURE = 0.03              # потолок норм. локальной дисперсии (тальк гладкий)
TALC_MIN_BLOB_AREA = 168             # px, отбрасывать мелкие крапинки
# Регион-контраст: тальк должен быть темнее ОКРУЖАЮЩЕГО гангу (а не просто тёмным).
# Отсекает плавные тёмные вариации нерудной матрицы (ложные срабатывания).
TALC_BG_WINDOW = 301                 # px, окно оценки фона по гангу (исключая тёмные кандидаты)
TALC_CONTRAST_MARGIN = 26.08          # тальк темнее окружающего гангу на столько интенсивностей
# ^ выше амплитуды яркостных вариаций гангу, но ниже контраста тусклого талька.
#   Ключевой калибровочный порог: ниже -> больше ложного талька, выше -> пропуски.

# Виньетка/чёрная рамка апертуры (отражённая микроскопия часто даёт круглое поле
# зрения с чёрными углами) — БЕЗ этого фильтра рамка ловится детектором как тальк
# (тёмная+гладкая+касается края), что раздувает talc_pct. Исключаем крупные тёмные
# области, СВЯЗАННЫЕ С КРАЕМ КАДРА, из кандидатов на тальк.
TALC_VIGNETTE_ABS_THR = 18.0         # 0-255; темнее этого + касается края -> виньетка, не тальк
TALC_VIGNETTE_MIN_AREA_FRAC = 0.003  # доля площади кадра, ниже которой не считаем виньеткой

# Тайлинг панорам (вход инференса): обрабатывать тайлами с перекрытием, не грузить 10k^2.
TILE_SIZE = 1024
TILE_OVERLAP = 128
MAX_PROCESS_SIDE = 8000              # выше — даунскейл (доли масштабно-инвариантны)
# TODO(P1): для многогигабайтных TIFF заменить даунскейл на ленивый тайлинг (pyvips/tifffile).

# Синие экспертные обводки (ground truth талька для калибровки) в HSV.
BLUE_HSV_LOWER = (90, 60, 40)
BLUE_HSV_UPPER = (140, 255, 255)

# --- Цвет маски талька (BGR для OpenCV) ------------------------------------
COLOR_TALC = (220, 0, 0)             # синий
MASK_ALPHA = 0.45                    # прозрачность наложения

# --- API / logging ---------------------------------------------------------
ANALYZE_LOG = LOG_DIR / "analyze.log"
