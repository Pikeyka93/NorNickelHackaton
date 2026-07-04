# ============================================================
# ВЛАДЕЛЕЦ: P1 (сегментация талька)
# TODO(P1): прогнать на РЕАЛЬНОЙ панораме на сервере, убедиться в бюджете ≤5 мин.
#           Если не укладывается / не хватает памяти — см. подсказку в выводе
#           (ленивый тайлинг pyvips/tifffile вместо cv2.imread + даунскейл).
# ============================================================
"""
Бенчмарк тайлинга: сколько времени и памяти требует segment_talc() на панораме.

Панорамы — рабочий вход инференса (по брифу — до ~10000x10000, файлы 47-212 МБ).
Сейчас `read_image_bgr` грузит файл ЦЕЛИКОМ через cv2.imread и только потом, если
сторона больше MAX_PROCESS_SIDE (config.py), даунскейлит — это НЕ ленивый тайлинг:
пиковая память = полный декодированный битмап. Для JPG/PNG панорам разумного размера
этого обычно достаточно; если файлы окажутся больше или это гигантские TIFF-стеки —
нужен pyvips (access="sequential") или tifffile, чтобы не грузить всё в память разом.

Использование:
    python scripts/benchmark_tiling.py /path/to/panorama.jpg
    python scripts/benchmark_tiling.py /path/to/panorama.tif --budget-sec 300
"""
from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

# make repo root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from core import segment  # noqa: E402


def _native_longest_side(path: Path) -> int:
    """Нативный размер файла без полного декодирования — PIL читает только заголовок."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return max(im.size)
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Бенчмарк тайлинга сегментации талька на панораме")
    ap.add_argument("image", help="путь к панораме (jpg/png/tif)")
    ap.add_argument("--budget-sec", type=float, default=300.0,
                    help="бюджет по брифу (по умолчанию 5 минут)")
    args = ap.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"[benchmark] файл не найден: {path}")
        sys.exit(1)
    file_mb = path.stat().st_size / (1024 * 1024)
    print(f"[benchmark] файл: {path.name} ({file_mb:.1f} МБ)")

    native_side = _native_longest_side(path)

    tracemalloc.start()
    t0 = time.time()
    bgr = segment.read_image_bgr(path)
    t_read = time.time() - t0
    h, w = bgr.shape[:2]
    downscaled = native_side > 0 and max(h, w) < native_side
    print(f"[benchmark] прочитано за {t_read:.1f}s -> {w}x{h}"
          + (f" (даунскейлено с нативных {native_side}px, MAX_PROCESS_SIDE={C.MAX_PROCESS_SIDE})"
             if downscaled else ""))

    single = (h <= C.TILE_SIZE and w <= C.TILE_SIZE)
    n_tiles = 1 if single else sum(
        1 for _ in segment.iter_tiles(h, w, C.TILE_SIZE, C.TILE_OVERLAP))
    print(f"[benchmark] TILE_SIZE={C.TILE_SIZE} TILE_OVERLAP={C.TILE_OVERLAP} -> {n_tiles} тайлов")

    t1 = time.time()
    talc = segment.segment_talc(bgr)
    t_seg = time.time() - t1
    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total = t_read + t_seg
    pct = segment.talc_percentage(talc)
    print(f"[benchmark] сегментация: {t_seg:.1f}s ({t_seg / max(n_tiles, 1):.2f}s/тайл)")
    print(f"[benchmark] talc_pct = {pct:.2f}%")
    print(f"[benchmark] пик Python-памяти (tracemalloc; не считает C-буферы cv2/numpy, "
          f"реальный пик RSS будет выше): {peak_mem / (1024 ** 2):.1f} МБ")
    status = "OK" if total <= args.budget_sec else "ПРЕВЫШЕН"
    print(f"[benchmark] ИТОГО: {total:.1f}s из бюджета {args.budget_sec:.0f}s ({status})")

    if total > args.budget_sec:
        print("[benchmark] Бюджет превышен. Варианты: увеличить TILE_SIZE (меньше тайлов, "
              "но крупнее окна TALC_LOCAL_WINDOW/TALC_BG_WINDOW относительно тайла), убрать "
              "лишние морф.операции, либо перейти на ленивое чтение (pyvips access='sequential' "
              "или tifffile) вместо полного cv2.imread + даунскейл.")
    if downscaled:
        print("[benchmark] Внимание: изображение даунскейлено ДО тайлинга. Доли талька "
              "масштабно-инвариантны, но самые тонкие прожилки у предела разрешения могут "
              "занижаться. Если это окажется критично на реальных панорамах — нужен ленивый "
              "тайлинг по нативному разрешению (см. TODO в config.py/segment.py).")


if __name__ == "__main__":
    main()
