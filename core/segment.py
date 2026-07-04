# ============================================================
# ВЛАДЕЛЕЦ: P1 (сегментация талька)
# TODO(P1):
#   - откалибровать TALC_* в config.py по синим обводкам (extract_blue_annotations),
#     цель: ошибка talc_pct ±3% (IoU/расхождение доли — см. notebooks/README.md);
#   - проверить бюджет тайлинга на реальной панораме (≤5 мин);
#   - при необходимости — ленивый тайлинг (pyvips/tifffile) вместо даунскейла.
# ============================================================
"""
Классический CV: сегментация ТОЛЬКО талька -> синяя маска поверх снимка + talc_pct.

По уточнениям жюри срастания НЕ сегментируем (разметки по ним нет). Единственная
пиксельная разметка — синие обводки оталькования, поэтому оценивается только тальк.

Пайплайн:
  1. per-image (per-tile) нормализация освещения   -- ОБЯЗАТЕЛЬНО первым шагом
  2. детекция талька: тёмная гладкая фаза, темнее локального фона, в нерудной матрице
  3. синяя маска талька поверх снимка + доля талька от всей площади
Панорамы (вход инференса) — тайлами с перекрытием.
`extract_blue_annotations` достаёт эталон талька из синих обводок для калибровки.
"""
from __future__ import annotations

from typing import Iterator, Union

import cv2
import numpy as np

import config as C

# skimage не обязателен: если нет — используем cv2-фолбэк локальной дисперсии.
try:
    from skimage.feature import local_binary_pattern  # noqa: F401
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except Exception:  # pragma: no cover
    Image = None

ImageInput = Union[str, "bytes", np.ndarray]
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_image_bgr(source: ImageInput, max_side: int = C.MAX_PROCESS_SIDE) -> np.ndarray:
    """Любой вход -> BGR uint8, с даунскейлом если сторона больше max_side (доли
    масштабно-инвариантны). TODO(P1): для многогигабайтных TIFF — ленивый тайлинг."""
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
            raise ValueError("Не удалось декодировать байты изображения.")
    else:  # path
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {source}")

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- #
# Шаг 1: нормализация освещения (per tile)
# --------------------------------------------------------------------------- #
def normalize_illumination(bgr: np.ndarray) -> np.ndarray:
    """Глобальная нормализация: gray-world баланс белого + стандартизация яркости.

    Убирает жёлтые/тёмные/светлые касты (снимки от почти чёрных до жёлтых), чтобы
    пороги текстуры/контраста были сопоставимы между кадрами.
    ВАЖНО: НЕ делаем пространственный flat-field (деление на размытый фон) — он
    принял бы КРУПНОЕ тёмное пятно талька за тень и «вычел» бы его. Локальную
    неравномерность освещения учитывает регион-контраст в detect_talc."""
    f = bgr.astype(np.float32)

    # 1) gray-world: каждый канал к общему среднему (убирает цветовой каст)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    f *= (means.mean() / means)

    # 2) стандартизация глобальной яркости: медиану -> 128 (сохраняет пространственную
    #    структуру, в т.ч. тёмный тальк — только выравнивает общий уровень)
    gray = f.mean(axis=2)
    med = float(np.median(gray)) + 1e-6
    f *= (128.0 / med)

    return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Шаг 2: детекция талька
# --------------------------------------------------------------------------- #
def _local_variance(gray_f: np.ndarray, win: int) -> np.ndarray:
    """Нормированная (0..1) локальная дисперсия через box-фильтры."""
    mean = cv2.boxFilter(gray_f, -1, (win, win))
    mean_sq = cv2.boxFilter(gray_f * gray_f, -1, (win, win))
    var = np.clip(mean_sq - mean * mean, 0, None)
    return var / (255.0 ** 2)


def _local_coherence(gray_f: np.ndarray, win: int) -> np.ndarray:
    """Когерентность локальной структуры (0..1) через тензор структуры (Sobel).

    0 = изотропно (нет выраженного направления градиента, как у рассеянного
    гладкого талька), 1 = сильно направленно (как у игольчатых/призматических
    зёрен сульфидов — они тёмные и гладкие ВНУТРИ зерна, поэтому проходят все
    остальные признаки талька, но имеют выраженную ось вытянутости, которой у
    талька нет). Используется как дополнительный фильтр ложных срабатываний на
    рядовых рудах с игольчатой текстурой (см. calibrate_talc.py / TALC_MAX_COHERENCE)."""
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.boxFilter(gx * gx, -1, (win, win))
    gyy = cv2.boxFilter(gy * gy, -1, (win, win))
    gxy = cv2.boxFilter(gx * gy, -1, (win, win))
    tmp = np.sqrt(np.clip((gxx - gyy) ** 2 + 4 * gxy ** 2, 0, None))
    l1 = (gxx + gyy + tmp) / 2
    l2 = (gxx + gyy - tmp) / 2
    return (l1 - l2) / (l1 + l2 + 1e-6)


def _vignette_mask(gray: np.ndarray) -> np.ndarray:
    """Маска чёрной рамки/виньетки апертуры, а НЕ талька.

    Отражённая оптическая микроскопия часто даёт круглое поле зрения с чёрными
    углами кадра (виньетка объектива); скриншоты/кропы иногда добавляют чёрные
    полосы у края. И то и другое — тёмное+гладкое+касается края кадра, то есть
    формально проходит все признаки талька из detect_talc(). Без явного исключения
    оно ошибочно посчиталось бы тальком и раздуло бы talc_pct. Здесь берём крупные
    тёмные связные области, касающиеся границы кадра, и трактуем их как "вне
    образца" (calibrate_talc.py сможет откалибровать TALC_VIGNETTE_* по факту)."""
    h, w = gray.shape
    dark = (gray < C.TALC_VIGNETTE_ABS_THR).astype(np.uint8)
    n, labels = cv2.connectedComponents(dark, 8)
    if n <= 1:
        return np.zeros((h, w), bool)
    border_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :]))
    border_labels |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    border_labels.discard(0)
    if not border_labels:
        return np.zeros((h, w), bool)
    min_area = C.TALC_VIGNETTE_MIN_AREA_FRAC * h * w
    mask = np.zeros((h, w), bool)
    for lbl in border_labels:
        comp = labels == lbl
        if comp.sum() >= min_area:
            mask |= comp
    return mask


def detect_talc(norm_bgr: np.ndarray) -> np.ndarray:
    """Бинарная маска талька (0/255).

    Тальк = пиксели, которые (a) не яркие (нерудная матрица), (d) низкой текстуры, и
    темнее фона: либо (b) по абсолюту (темнее уровня матрицы на TALC_ABS_MARGIN —
    ловит ИНТЕРЬЕР больших зон, т.к. после нормализации фон почти ровный), либо
    (c) темнее локального фона на TALC_LOCAL_MARGIN (края / остаточный градиент).
    Чёрная рамка/виньетка апертуры исключается отдельно (см. _vignette_mask) —
    иначе она проходит все признаки талька и раздувает долю. На равномерном фоне
    без талька результат ~пустой (не раздуваем долю). Калибровать по синим обводкам."""
    gray = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = C.TALC_LOCAL_WINDOW | 1

    vignette = _vignette_mask(gray)                  # рамка/виньетка — не образец

    bright_cut = np.percentile(gray, C.TALC_BRIGHT_EXCLUDE_PERCENTILE)
    matrix = (gray < bright_cut) & ~vignette          # (a) нерудная матрица, без рамки
    if matrix.sum() < 10:
        return np.zeros(gray.shape, np.uint8)

    bg = np.percentile(gray[matrix], C.TALC_DARK_PERCENTILE)    # уровень нерудного фона
    dark_abs = gray < (bg - C.TALC_ABS_MARGIN)                  # (b) абсолютно темнее фона
    local_mean = cv2.boxFilter(gray, -1, (win, win))
    dark_local = gray < (local_mean - C.TALC_LOCAL_MARGIN)      # (c) темнее локального фона
    smooth = _local_variance(gray, win) < C.TALC_MAX_TEXTURE    # (d) гладкий
    cand = matrix & smooth & (dark_abs | dark_local)

    # (d2) изотропность: тальк рассеян и не имеет выраженной оси, а игольчатые
    # сульфиды — тёмные+гладкие ВНУТРИ зерна, но вытянутые -> высокая когерентность.
    # Без этого фильтра плотные скопления игольчатых зёрен на рядовой руде ошибочно
    # ловятся как тальк (см. коммит с "needle-grain false positive").
    isotropic = _local_coherence(gray, win) < C.TALC_MAX_COHERENCE
    cand = cand & isotropic

    # (e) регион-контраст: оценить фон по ОКРУЖАЮЩЕМУ гангу (исключая тёмные кандидаты)
    # и оставить только пиксели, что темнее этого фона -> отсекает плавные тёмные
    # вариации матрицы, сохраняя интерьер настоящих тальк-зон.
    win2 = C.TALC_BG_WINDOW | 1
    gangue = ((~cand) & matrix).astype(np.float32)
    gsum = cv2.boxFilter(gray * gangue, -1, (win2, win2), normalize=False)
    gcnt = cv2.boxFilter(gangue, -1, (win2, win2), normalize=False)
    bg_local = gsum / (gcnt + 1e-6)
    contrast_ok = (gcnt > 0) & (gray < bg_local - C.TALC_CONTRAST_MARGIN)

    talc = (cand & contrast_ok & ~vignette).astype(np.uint8) * 255
    talc = cv2.morphologyEx(talc, cv2.MORPH_OPEN, _K5)
    # Укрупнение формы: реальный тальк — большие сплошные пятна (см. эталонные
    # обводки), а не пиксельная рябь. Раньше closing был фиксирован на 5px, из-за
    # чего кандидаты (даже верные, внутри настоящих тальк-зон) оставались
    # рассыпанными точками -> IoU с эталоном был ~0.05-0.07 при формально похожем
    # проценте площади (ложный шум в других местах кадра компенсировал недобор
    # внутри реальных зон). TALC_MORPH_CLOSE_KERNEL мержит соседние кандидаты в
    # сплошные пятна; TALC_MIN_BLOB_AREA после этого отсекает то, что не срослось
    # в пятно нужного масштаба (шум), а не мелкие внутренние вкрапления талька.
    close_k = C.TALC_MORPH_CLOSE_KERNEL | 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    talc = cv2.morphologyEx(talc, cv2.MORPH_CLOSE, close_kernel)
    return _drop_small(talc, C.TALC_MIN_BLOB_AREA)


# --------------------------------------------------------------------------- #
# Синие экспертные обводки (эталон для калибровки)
# --------------------------------------------------------------------------- #
def extract_blue_annotations(bgr: np.ndarray, fill: bool = True) -> np.ndarray:
    """Маска синих обводок талька (ground truth). При fill — заливка контура, чтобы
    получить размеченную ОБЛАСТЬ, а не только линию. Обводки полные (жюри)."""
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
# Тайлинг (панорамы на инференсе)
# --------------------------------------------------------------------------- #
def iter_tiles(h: int, w: int, tile: int, overlap: int) -> Iterator[tuple[int, int, int, int]]:
    """Перекрывающиеся тайлы (y0, y1, x0, x1), покрывающие h*w."""
    step = max(1, tile - overlap)
    for y0 in range(0, max(1, h - overlap), step):
        y1 = min(h, y0 + tile)
        for x0 in range(0, max(1, w - overlap), step):
            x1 = min(w, x0 + tile)
            yield y0, y1, x0, x1


def segment_talc(bgr: np.ndarray) -> np.ndarray:
    """Полноразмерная маска талька: per-tile нормализация + детекция, сшивка объединением."""
    h, w = bgr.shape[:2]
    talc = np.zeros((h, w), np.uint8)
    single = (h <= C.TILE_SIZE and w <= C.TILE_SIZE)
    tiles = [(0, h, 0, w)] if single else iter_tiles(h, w, C.TILE_SIZE, C.TILE_OVERLAP)
    for (y0, y1, x0, x1) in tiles:
        norm = normalize_illumination(bgr[y0:y1, x0:x1])
        t = detect_talc(norm)
        talc[y0:y1, x0:x1] = np.maximum(talc[y0:y1, x0:x1], t)
    return talc


# --------------------------------------------------------------------------- #
# Наложение + доля
# --------------------------------------------------------------------------- #
def build_talc_overlay(bgr: np.ndarray, talc_mask: np.ndarray,
                       alpha: float = C.MASK_ALPHA) -> np.ndarray:
    """Синяя маска талька поверх снимка (BGR) — контур линией + лёгкая заливка,
    как в экспертной разметке (обводка, а не сплошная закраска). Сплошная заливка
    визуально сливалась в кляксы и не читалась как "область талька" — обводка
    линией даёт тот же визуальный язык, что и эталон (см. Области оталькования)."""
    out = bgr.copy()

    # лёгкая заливка (по брифу нужна маска; делаем её едва заметной, не основной)
    if alpha > 0:
        blue = np.zeros_like(bgr)
        blue[:] = C.COLOR_TALC
        m = talc_mask > 0
        out[m] = cv2.addWeighted(bgr, 1 - alpha, blue, alpha, 0)[m]

    # контур поверх заливки — основной визуальный акцент, как у эксперта.
    # RETR_EXTERNAL (не RETR_LIST): только внешняя граница каждого пятна, без
    # внутренних дырок-контуров — эксперт тоже обводит ПО ВНЕШНЕМУ КОНТУРУ зоны,
    # не вырезая мелкие внутренние вкрапления отдельными линиями.
    contours, _ = cv2.findContours((talc_mask > 0).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(2, round(0.002 * max(bgr.shape[:2])))
    cv2.drawContours(out, contours, -1, C.COLOR_TALC, thickness, lineType=cv2.LINE_AA)
    return out


def talc_percentage(talc_mask: np.ndarray) -> float:
    """Доля талька от ВСЕЙ площади шлифа, %."""
    total = talc_mask.shape[0] * talc_mask.shape[1]
    return round(100.0 * int((talc_mask > 0).sum()) / max(total, 1), 2)


def analyze_image(source: ImageInput) -> dict:
    """Полная тальк-сегментация. Возвращает bgr / talc_pct / talc_mask / overlay (массивы;
    base64-кодирование — в analyze())."""
    bgr = read_image_bgr(source)
    talc_mask = segment_talc(bgr)
    return {
        "bgr": bgr,
        "talc_pct": talc_percentage(talc_mask),
        "talc_mask": talc_mask,
        "overlay": build_talc_overlay(bgr, talc_mask),
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
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
        print("talc_pct:", analyze_image(sys.argv[1])["talc_pct"])
