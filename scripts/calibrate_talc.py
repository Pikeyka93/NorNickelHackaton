# ============================================================
# ВЛАДЕЛЕЦ: P1 (сегментация талька)
# TODO(P1): прогнать на сервере против DATA_DIR, применить лучшие параметры
#           (--apply) в config.py. Цель из брифа: mean abs talc_pct error <= 3%.
# ============================================================
"""
Калибровка порогов TALC_* (config.py) по экспертной разметке талька.

Разметка в датасете — не синие линии поверх самого снимка, а ОТДЕЛЬНАЯ подпапка
рядом с оригиналами (по факту: `<класс>/Области оталькования/<то же имя файла>`,
см. debug-скрипт коллеги). Внутри неё файл может быть либо копией снимка с синим
контуром талька, либо готовой маской — детектируем автоматически по количеству
синих пикселей (`core.segment.extract_blue_annotations`). Скрипт:

  1. находит пары (оригинал, эталон) — по подпапкам, чьё имя похоже на
     "Области оталькования" (гибко, т.к. ч1/ч2 могут называть по-разному), плюс
     резервный вариант: синие обводки прямо на самом оригинале;
  2. для каждой пары считает эталонную маску/долю талька;
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config as C  # noqa: E402
from core import segment  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Ключевые слова в имени подпапки, которая содержит экспертную разметку талька
# рядом с оригиналами (по факту в датасете: "Области оталькования"). Гибко на
# случай, если ч1/ч2 называют её чуть иначе.
EXPERT_SUBDIR_KEYWORDS = ("облас", "тальк", "оталь")
MIN_BLUE_PIXELS = 200  # порог "заметных" синих пикселей, иначе считаем шумом

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
    orig_path: Path
    expert_path: Path
    bgr: "np.ndarray"
    gt_mask: "np.ndarray"
    gt_pct: float
    gt_kind: str  # "blue_outline" | "direct_mask" — для отчёта/отладки


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #
def _is_expert_subdir(name: str) -> bool:
    key = name.strip().lower()
    return any(kw in key for kw in EXPERT_SUBDIR_KEYWORDS)


def find_annotated_pairs(data_dir: Path, limit: Optional[int] = None) -> list[tuple[Path, Path]]:
    """Найти пары (оригинал, файл-эталон талька).

    Основной путь: подпапки вида `<класс>/Области оталькования/<то же имя>` —
    именно так устроена разметка в датасете (см. debug-скрипт коллеги: EXPERT_DIR
    = ORIG_DIR / "Области оталькования"). Резервный путь (на случай другого
    расположения в ч2 или ручных тестов): синие обводки прямо на самом снимке.
    """
    pairs: list[tuple[Path, Path]] = []
    seen_orig: set[Path] = set()

    # --- путь 1: сиблинг-подпапка с эталонами ---
    for expert_dir in sorted(data_dir.rglob("*")):
        if not expert_dir.is_dir() or not _is_expert_subdir(expert_dir.name):
            continue
        orig_dir = expert_dir.parent
        for expert_path in sorted(expert_dir.iterdir()):
            if not expert_path.is_file() or expert_path.suffix.lower() not in IMG_EXTS:
                continue
            orig_path = orig_dir / expert_path.name
            if orig_path.exists() and orig_path not in seen_orig:
                pairs.append((orig_path, expert_path))
                seen_orig.add(orig_path)
                if limit and len(pairs) >= limit:
                    return pairs

    # --- путь 2 (резерв): синие обводки прямо на оригинале ---
    if not limit or len(pairs) < limit:
        for p in sorted(data_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS or p in seen_orig:
                continue
            if any(_is_expert_subdir(parent.name) for parent in p.parents):
                continue  # сами файлы внутри эталонных подпапок пропускаем
            try:
                bgr = segment.read_image_bgr(p)
            except Exception:
                continue
            blue = segment.extract_blue_annotations(bgr, fill=False)
            if int((blue > 0).sum()) < MIN_BLUE_PIXELS:
                continue
            pairs.append((p, p))  # эталон и оригинал — один и тот же файл
            seen_orig.add(p)
            if limit and len(pairs) >= limit:
                break

    return pairs


def _ground_truth_from_expert(expert_bgr: "np.ndarray") -> tuple["np.ndarray", str]:
    """Эталонная маска талька из файла-эталона + пометка, как она получена.

    Файл-эталон бывает либо копией снимка с синим контуром талька (тогда достаём
    `extract_blue_annotations`), либо уже готовой маской (бинарной/почти бинарной
    заливкой) — тогда просто бинаризуем. Определяем по количеству синих пикселей."""
    blue = segment.extract_blue_annotations(expert_bgr, fill=False)
    if int((blue > 0).sum()) >= MIN_BLUE_PIXELS:
        return segment.extract_blue_annotations(expert_bgr, fill=True), "blue_outline"

    gray = cv2.cvtColor(expert_bgr, cv2.COLOR_BGR2GRAY)
    # Готовая маска обычно почти бинарна (мало уникальных значений) — Otsu ок в обоих случаях.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Если после Otsu "тальком" оказалось больше половины кадра — скорее всего инверсия
    # (маска нарисована в 0=тальк) или это не маска вовсе; берём меньшую по площади сторону.
    if (mask > 0).mean() > 0.5:
        mask = 255 - mask
    return mask, "direct_mask"


def load_calib_set(pairs: list[tuple[Path, Path]]) -> list[CalibSample]:
    samples = []
    for orig_path, expert_path in pairs:
        try:
            bgr = segment.read_image_bgr(orig_path)
            expert_bgr = (bgr if expert_path == orig_path
                          else segment.read_image_bgr(expert_path))
        except Exception as e:
            print(f"  [!] пропускаю {orig_path.name}: {e}")
            continue
        gt, kind = _ground_truth_from_expert(expert_bgr)
        if gt.shape[:2] != bgr.shape[:2]:
            gt = cv2.resize(gt, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        samples.append(CalibSample(orig_path=orig_path, expert_path=expert_path, bgr=bgr,
                                    gt_mask=gt, gt_pct=segment.talc_percentage(gt), gt_kind=kind))
    return samples


# --------------------------------------------------------------------------- #
# Поиск параметров
# --------------------------------------------------------------------------- #
def _set_params(params: dict) -> None:
    for k, v in params.items():
        setattr(C, k, v)


def evaluate(params: dict, samples: list[CalibSample]) -> dict:
    """Гоняет ТОЧНО тот же путь, что и продакшен — segment.segment_talc(bgr)
    (тайлинг + per-tile нормализация), а не укороченную "как в analyze_image"
    версию. На снимках в датасете (крупнее TILE_SIZE) тайлинг эмпирически стоит
    примерно столько же, сколько нормализация всего снимка целиком — так что
    упрощаем и везде считаем честно, без риска разъехаться с реальным инференсом."""
    _set_params(params)
    errs, ious = [], []
    for s in samples:
        pred = segment.segment_talc(s.bgr)
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
    ap = argparse.ArgumentParser(description="Калибровка TALC_* по экспертной разметке талька")
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--max-images", type=int, default=60,
                    help="сколько размеченных пар (оригинал+эталон) использовать")
    ap.add_argument("--trials", type=int, default=200, help="случайных комбинаций перебрать")
    ap.add_argument("--refine-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--apply", action="store_true",
                    help="переписать TALC_* в config.py лучшими найденными значениями")
    ap.add_argument("--report", default=str(C.REPORTS_DIR / "talc_calibration.json"))
    ap.add_argument("--resume-from", default=None,
                    help="JSON-отчёт предыдущего запуска (--report) — продолжить поиск "
                         "от его best.params вместо текущих значений config.py. Удобно "
                         "дробить долгий поиск на несколько коротких запусков подряд.")
    ap.add_argument("--time-budget-sec", type=float, default=None,
                    help="остановить перебор (не начинать новую попытку), если вышло время — "
                         "для запуска короткими кусками под ограничение по времени вызова")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[calibrate_talc] ищу пары (оригинал, эталон талька) в {data_dir} ...")
    pairs = find_annotated_pairs(data_dir, limit=args.max_images)
    if not pairs:
        print("[calibrate_talc] не нашёл ни одной размеченной пары — проверь --data-dir, "
              f"имя подпапки с эталонами (ищу по ключевым словам {EXPERT_SUBDIR_KEYWORDS}) "
              "и BLUE_HSV_LOWER/UPPER в config.py.")
        sys.exit(1)
    print(f"[calibrate_talc] калибровочный набор: {len(pairs)} пар")
    samples = load_calib_set(pairs)
    if not samples:
        print("[calibrate_talc] пары нашлись, но ни одна не прочиталась — см. сообщения выше.")
        sys.exit(1)
    n_blue = sum(1 for s in samples if s.gt_kind == "blue_outline")
    n_mask = sum(1 for s in samples if s.gt_kind == "direct_mask")
    print(f"[calibrate_talc] эталон: {n_blue} через синие обводки, {n_mask} как готовая маска")

    baseline_params = {k: getattr(C, k) for k in PARAM_SPACE}
    baseline_metrics = evaluate(baseline_params, samples)
    print(f"[calibrate_talc] базовые (текущие) параметры: {baseline_metrics}")

    # --resume-from: продолжаем поиск от лучшей точки предыдущего запуска, а не от
    # config.py — так можно дробить долгий перебор на много коротких запусков подряд.
    start_params, start_metrics = baseline_params, baseline_metrics
    if args.resume_from:
        prev = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))
        start_params = prev["best"]["params"]
        start_metrics = evaluate(start_params, samples)  # пересчитываем на ТЕКУЩЕЙ выборке
        print(f"[calibrate_talc] продолжаю от {args.resume_from}: {start_metrics}")

    rng = random.Random(args.seed)
    best_params, best_metrics = start_params, start_metrics
    deadline = (time.time() + args.time_budget_sec) if args.time_budget_sec else None
    t0 = time.time()
    trials_done = 0
    for i in range(args.trials):
        if deadline and time.time() >= deadline:
            print(f"  [{i:4d}] закончилось время (--time-budget-sec) — останавливаю перебор")
            break
        trial = random_params(rng)
        m = evaluate(trial, samples)
        trials_done += 1
        if score(m) < score(best_metrics):
            best_params, best_metrics = trial, m
            print(f"  [{i:4d}] new best: mean_err={m['mean_abs_err_pct']:.2f}% "
                  f"iou={m['mean_iou']:.3f}")
    refine_steps = args.refine_steps
    if deadline:
        remaining = deadline - time.time()
        per_eval = (time.time() - t0) / max(trials_done, 1)
        refine_steps = max(0, min(refine_steps, int(remaining / max(per_eval, 0.01))))
    best_params, best_metrics = coordinate_refine(
        best_params, best_metrics, samples, rng, steps=refine_steps)
    dt = time.time() - t0
    print(f"[calibrate_talc] сделано: {trials_done} случайных попыток + {refine_steps} шагов уточнения")

    print(f"\n[calibrate_talc] готово за {dt:.1f}s (честная оценка, с тайлингом как в проде). Лучшее:")
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
