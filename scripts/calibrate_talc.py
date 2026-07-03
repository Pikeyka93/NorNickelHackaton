# ============================================================
# ВЛАДЕЛЕЦ: P1 (сегментация талька)
# TODO(P1): прогнать на сервере против DATA_DIR (там снимки с синими обводками),
#           применить лучшие параметры (--apply) в config.py.
#           Цель из брифа: mean abs talc_pct error <= 3%.
# ============================================================
"""
Калибровка порогов TALC_* (config.py) по синим экспертным обводкам.

Единственная пиксельная разметка в датасете — синие контуры талька, нарисованные
экспертом поверх части оталькованных снимков. Скрипт:

  1. находит снимки с заметными синими обводками под --data-dir
     (`core.segment.extract_blue_annotations`);
  2. для каждого считает эталон: маску талька и её долю (%) от площади снимка;
  3. случайным поиском + локальным уточнением подбирает TALC_* так, чтобы
     `detect_talc()` был ближе всего к эталону (главный критерий — |Δ talc_pct|,
     цель <=3%; IoU — вторичный сигнал, обводки размечают не каждое пятнышко талька);
  4. печатает лучшую комбинацию и пишет отчёт (JSON); с --apply переписывает
     TALC_* прямо в config.py (точечная замена, остальной файл не трогает).

Запуск (нужны данные — на сервере):
    python scripts/calibrate_talc.py --data-dir "$DATA_DIR" --trials 300
    python scripts/calibrate_talc.py --apply              # переписать config.py

Без датасета скрипт ничего не откалибрует (упадёт с понятным сообщением) — это
ожидаемо, писался без доступа к данным. Логика проверена на синтетике при разработке.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# make repo root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config as C  # noqa: E402
from core import segment  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Параметры, которые крутим, и диапазоны поиска (см. TODO(P1) в config.py).
PARAM_SPACE: dict[str, tuple[float, float]] = {
    "TALC_BRIGHT_EXCLUDE_PERCENTILE": (70, 95),
    "TALC_DARK_PERCENTILE": (30, 70),
    "TALC_ABS_MARGIN": (10.0, 40.0),
    "TALC_LOCAL_MARGIN": (3.0, 15.0),
    "TALC_MAX_TEXTURE": (0.03, 0.15),
    "TALC_CONTRAST_MARGIN": (10.0, 35.0),
    "TALC_MIN_BLOB_AREA": (50, 400),
}
INT_PARAMS = {"TALC_BRIGHT_EXCLUDE_PERCENTILE", "TALC_DARK_PERCENTILE", "TALC_MIN_BLOB_AREA"}


@dataclass
class CalibSample:
    path: Path
    bgr: "np.ndarray"
    gt_mask: "np.ndarray"
    gt_pct: float


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #
def find_annotated_images(data_dir: Path, limit: Optional[int] = None) -> list[Path]:
    """Найти снимки с заметными синими обводками (кандидаты для калибровки)."""
    found: list[Path] = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        try:
            bgr = segment.read_image_bgr(p)
        except Exception:
            continue
        blue = segment.extract_blue_annotations(bgr, fill=False)
        if int((blue > 0).sum()) < 200:  # шум / случайные синие пиксели отсекаем
            continue
        found.append(p)
        if limit and len(found) >= limit:
            break
    return found


def load_calib_set(paths: list[Path]) -> list[CalibSample]:
    samples = []
    for p in paths:
        bgr = segment.read_image_bgr(p)
        gt = segment.extract_blue_annotations(bgr, fill=True)
        samples.append(CalibSample(path=p, bgr=bgr, gt_mask=gt,
                                    gt_pct=segment.talc_percentage(gt)))
    return samples


# --------------------------------------------------------------------------- #
# Поиск параметров
# --------------------------------------------------------------------------- #
def _set_params(params: dict) -> None:
    for k, v in params.items():
        setattr(C, k, v)


def evaluate(params: dict, samples: list[CalibSample]) -> dict:
    """Прогнать detect_talc с данными params по всем сэмплам, вернуть метрики."""
    _set_params(params)
    errs, ious = [], []
    for s in samples:
        norm = segment.normalize_illumination(s.bgr)
        pred = segment.detect_talc(norm)
        pred_pct = segment.talc_percentage(pred)
        errs.append(abs(pred_pct - s.gt_pct))
        inter = int(((pred > 0) & (s.gt_mask > 0)).sum())
        union = int(((pred > 0) | (s.gt_mask > 0)).sum())
        ious.append(inter / union if union else 1.0)
    return {
        "mean_abs_err_pct": float(np.mean(errs)),
        "max_abs_err_pct": float(np.max(errs)),
        "mean_iou": float(np.mean(ious)),
    }


def score(metrics: dict) -> float:
    """Ниже — лучше. Главная цель брифа: |Δtalc_pct| <= 3%; IoU — тай-брейк."""
    return metrics["mean_abs_err_pct"] + (1.0 - metrics["mean_iou"]) * 2.0


def random_params(rng: random.Random) -> dict:
    out = {}
    for name, (lo, hi) in PARAM_SPACE.items():
        v = rng.uniform(lo, hi)
        out[name] = int(round(v)) if name in INT_PARAMS else round(v, 2)
    return out


def coordinate_refine(best_params: dict, best_metrics: dict, samples: list[CalibSample],
                       rng: random.Random, steps: int) -> tuple[dict, dict]:
    """Локальное уточнение вокруг лучшей точки: один параметр за шаг."""
    for _ in range(steps):
        name = rng.choice(list(PARAM_SPACE.keys()))
        lo, hi = PARAM_SPACE[name]
        span = hi - lo
        cand = best_params[name] + span * rng.uniform(-0.15, 0.15)
        cand = max(lo, min(hi, cand))
        cand = int(round(cand)) if name in INT_PARAMS else round(cand, 2)
        trial = dict(best_params)
        trial[name] = cand
        m = evaluate(trial, samples)
        if score(m) < score(best_metrics):
            best_params, best_metrics = trial, m
    return best_params, best_metrics


# --------------------------------------------------------------------------- #
# Запись лучших значений в config.py
# --------------------------------------------------------------------------- #
def _apply_to_config_file(params: dict) -> None:
    """Точечно заменить значения TALC_* в config.py (не трогая остальной файл)."""
    cfg_path = Path(__file__).resolve().parent.parent / "config.py"
    text = cfg_path.read_text(encoding="utf-8")
    for name, value in params.items():
        pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)[^\s#]+", re.MULTILINE)
        if not pattern.search(text):
            print(f"  [!] не нашёл {name} в config.py — пропускаю")
            continue
        text = pattern.sub(lambda m, v=value: f"{m.group(1)}{v}", text, count=1)
    cfg_path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Калибровка TALC_* по синим экспертным обводкам")
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--max-images", type=int, default=60,
                    help="сколько размеченных (с синими обводками) снимков использовать")
    ap.add_argument("--trials", type=int, default=200, help="случайных комбинаций перебрать")
    ap.add_argument("--refine-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--apply", action="store_true",
                    help="переписать TALC_* в config.py лучшими найденными значениями")
    ap.add_argument("--report", default=str(C.REPORTS_DIR / "talc_calibration.json"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[calibrate_talc] ищу снимки с синими обводками в {data_dir} ...")
    paths = find_annotated_images(data_dir, limit=args.max_images)
    if not paths:
        print("[calibrate_talc] не нашёл ни одного снимка с синими обводками — "
              "проверь --data-dir и BLUE_HSV_LOWER/UPPER в config.py.")
        sys.exit(1)
    print(f"[calibrate_talc] калибровочный набор: {len(paths)} снимков")
    samples = load_calib_set(paths)

    baseline_params = {k: getattr(C, k) for k in PARAM_SPACE}
    baseline_metrics = evaluate(baseline_params, samples)
    print(f"[calibrate_talc] базовые (текущие) параметры: {baseline_metrics}")

    rng = random.Random(args.seed)
    best_params, best_metrics = baseline_params, baseline_metrics
    t0 = time.time()
    for i in range(args.trials):
        trial = random_params(rng)
        m = evaluate(trial, samples)
        if score(m) < score(best_metrics):
            best_params, best_metrics = trial, m
            print(f"  [{i:4d}] new best: mean_err={m['mean_abs_err_pct']:.2f}% "
                  f"iou={m['mean_iou']:.3f}")
    best_params, best_metrics = coordinate_refine(
        best_params, best_metrics, samples, rng, steps=args.refine_steps)
    dt = time.time() - t0

    print(f"\n[calibrate_talc] готово за {dt:.1f}s. Лучшее:")
    print(f"  mean_abs_err_pct = {best_metrics['mean_abs_err_pct']:.2f}%  "
          f"(цель <= 3%)   mean_iou = {best_metrics['mean_iou']:.3f}")
    for k, v in best_params.items():
        print(f"  {k} = {v}")

    report = {
        "data_dir": str(data_dir),
        "n_images": len(samples),
        "baseline": {"params": baseline_params, "metrics": baseline_metrics},
        "best": {"params": best_params, "metrics": best_metrics},
        "trials": args.trials,
        "refine_steps": args.refine_steps,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[calibrate_talc] отчёт сохранён: {args.report}")

    if args.apply:
        _apply_to_config_file(best_params)
        print("[calibrate_talc] config.py обновлён новыми TALC_* значениями.")
    else:
        print("[calibrate_talc] запусти с --apply, чтобы записать эти значения в config.py.")


if __name__ == "__main__":
    main()
