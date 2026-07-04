# ============================================================
# ВЛАДЕЛЕЦ: P1 (сегментация талька)
# СТАТУС: ЭКСПЕРИМЕНТ, НЕ ПОДКЛЮЧЁН К ПРОДАКШЕНУ (core/segment.py его не
# использует). Идея была хорошая — обучаемый по-пиксельный классификатор
# вместо ручной AND-цепочки порогов, которая упёрлась в потолок по форме
# маски (см. историю коммитов). НО по факту:
#   - held-out pixel AUC всего ~0.73-0.75 (на СБАЛАНСИРОВАННОЙ выборке) —
#     умеренно, не сильно;
#   - на ПОЛНОМ снимке (реальный дисбаланс классов) оптимальный порог
#     вероятности крайне нестабилен и сильно пляшет между конфигами обучения
#     (то заливает 40-60% кадра при thr=0.5, то почти ничего не находит) —
#     лучший найденный IoU на held-out ~0.15-0.17, то есть НЕ лучше и не
#     стабильнее классического detect_talc (mean_iou 0.070 на всех 42 парах,
#     проверено честно с тайлингом);
#   - 42 картинки — очень маленький датасет для generalizable порога.
# Вывод: набор признаков (яркость на 5 масштабах + текстура + когерентность +
# насыщенность/тон + разрыв до уровня матрицы) сам по себе не даёт решающего
# преимущества над руками откалиброванными порогами. Если возвращаться к этой
# идее — нужны либо признаки получше (не просто классические CV-каналы),
# либо больше размеченных данных, либо честная калибровка порога вероятности
# (Platt/isotonic) по реальному классовому дисбалансу, а не по искусственно
# сбалансированной обучающей выборке.
# ============================================================
"""
Обучение по-пиксельного классификатора талька (RandomForest, sklearn) на 42
парах (оригинал + синяя экспертная обводка) из "Оталькованные руды/Области
оталькования". Признаки — классические CV-каналы (яркость, локальный контраст
на нескольких масштабах, текстура, когерентность структуры, насыщенность/тон
цвета), не сырые пиксели — так модель мала, быстро обучается на CPU и не
переобучается на 42 картинках.

Разбиение на train/test — ПО СНИМКАМ (не по пикселям), иначе пиксели одного
и того же снимка утекут между train и test и оценка будет врать (пиксели
внутри одного пятна талька почти идентичны соседям).

Запуск (нужны данные — на сервере или локальной копии):
    python scripts/train_talc_segmenter.py --data-dir "$DATA_DIR"
    -> сохраняет weights/talc_rf.pkl (модель + список признаков + порог)

core/segment.py::detect_talc_ml() грузит эту модель ЛЕНИВО (один раз при
первом вызове) и используется в detect_talc(), ЕСЛИ файл весов существует —
иначе тихий fallback на классический порогово-эвристический путь (чтобы
свежий checkout без обученных весов не падал, аналогично core/classifier.py).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config as C  # noqa: E402
from core import segment  # noqa: E402
from scripts.calibrate_talc import find_annotated_pairs, load_calib_set  # noqa: E402

FEATURE_WINDOWS = (41, 101, 201, 401, 601)
COH_TEXTURE_WINDOW = 101
FEATURE_NAMES = [
    "gray", "gray-lm41", "gray-lm101", "gray-lm201", "gray-lm401", "gray-lm601",
    "var101", "coh101", "sat", "sat_local101", "hue", "bright_pctl_gap",
]


def extract_features(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H,W,K) float32 признаков + (H,W) bool маска виньетки (исключить из выборки)."""
    norm = segment.normalize_illumination(bgr)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, _ = cv2.split(hsv)

    w1, w2, w3, w4, w5 = FEATURE_WINDOWS
    lm1 = cv2.boxFilter(gray, -1, (w1, w1))
    lm2 = cv2.boxFilter(gray, -1, (w2, w2))
    lm3 = cv2.boxFilter(gray, -1, (w3, w3))
    lm4 = cv2.boxFilter(gray, -1, (w4, w4))
    lm5 = cv2.boxFilter(gray, -1, (w5, w5))
    var2 = segment._local_variance(gray, COH_TEXTURE_WINDOW)
    coh2 = segment._local_coherence(gray, COH_TEXTURE_WINDOW)
    s_local = cv2.boxFilter(s_ch, -1, (w2, w2))
    # разрыв до "уровня матрицы" (как в классическом detect_talc: bg по перцентилю
    # немаскированных пикселей) — самый сильный признак в ручном пороговом подходе,
    # даём модели его напрямую, а не только локальный контраст.
    bright_cut = np.percentile(gray, C.TALC_BRIGHT_EXCLUDE_PERCENTILE)
    matrix = gray < bright_cut
    bg = np.percentile(gray[matrix], C.TALC_DARK_PERCENTILE) if matrix.sum() > 10 else float(np.median(gray))
    bright_gap = gray - bg

    feats = np.stack(
        [gray, gray - lm1, gray - lm2, gray - lm3, gray - lm4, gray - lm5,
         var2, coh2, s_ch, s_local, h_ch, bright_gap],
        axis=-1,
    ).astype(np.float32)
    vignette = segment._vignette_mask(gray)
    return feats, vignette


def build_dataset(samples, rng: np.random.Generator, pos_cap: int = 6000, neg_ratio: float = 8.0):
    """Возвращает X (N,K), y (N,), groups (N,) — groups = индекс снимка (для сплита)."""
    xs, ys, groups = [], [], []
    for gi, s in enumerate(samples):
        feats, vignette = extract_features(s.bgr)
        gt = (s.gt_mask > 0) & ~vignette
        valid = ~vignette
        h, w = gt.shape

        pos_idx = np.flatnonzero(gt.ravel())
        neg_idx = np.flatnonzero((valid & ~gt).ravel())
        if len(pos_idx) == 0:
            continue
        if len(pos_idx) > pos_cap:
            pos_idx = rng.choice(pos_idx, pos_cap, replace=False)
        n_neg = min(len(neg_idx), int(len(pos_idx) * neg_ratio))
        neg_idx = rng.choice(neg_idx, n_neg, replace=False)

        idx = np.concatenate([pos_idx, neg_idx])
        flat = feats.reshape(-1, feats.shape[-1])
        xs.append(flat[idx])
        ys.append(np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))]))
        groups.append(np.full(len(idx), gi))
        print(f"  [{gi:2d}] {s.orig_path.name}: pos={len(pos_idx)} neg={n_neg}", flush=True)

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.concatenate(groups, axis=0)
    return X, y, g


def predict_mask(clf, bgr: np.ndarray, threshold: float = 0.5,
                  min_blob_area: int = 400, close_kernel: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Полная реконструкция маски по всем пикселям снимка (для честного сравнения
    с классическим detect_talc — та же морфологическая чистка на выходе)."""
    feats, vignette = extract_features(bgr)
    h, w, k = feats.shape
    proba = clf.predict_proba(feats.reshape(-1, k))[:, 1].reshape(h, w)
    mask = ((proba >= threshold) & ~vignette).astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
    ck = close_kernel | 1
    kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose)
    mask = segment._drop_small(mask, min_blob_area)
    return mask, proba


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DATA_DIR / C.PART1_DIR / "Оталькованные руды"))
    ap.add_argument("--max-images", type=int, default=60)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--out", default=str(C.WEIGHTS_DIR / "talc_rf.pkl"))
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    data_dir = Path(args.data_dir)
    pairs = find_annotated_pairs(data_dir, limit=args.max_images)
    print(f"[train_talc_segmenter] {len(pairs)} пар в {data_dir}")
    samples = load_calib_set(pairs)
    print(f"[train_talc_segmenter] загружено {len(samples)} снимков")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(samples))
    n_test = max(1, int(len(samples) * args.test_frac))
    test_ids = set(order[:n_test].tolist())
    train_samples = [s for i, s in enumerate(samples) if i not in test_ids]
    test_samples = [s for i, s in enumerate(samples) if i in test_ids]
    print(f"[train_talc_segmenter] train={len(train_samples)} снимков, "
          f"test={len(test_samples)} снимков (сплит ПО СНИМКАМ)")

    t0 = time.time()
    X_train, y_train, _ = build_dataset(train_samples, rng)
    print(f"[train_talc_segmenter] обучающая выборка: {X_train.shape}, "
          f"talc={int(y_train.sum())}/{len(y_train)} ({time.time()-t0:.1f}s)")

    # HistGradientBoosting: заметно быстрее RandomForest на CPU при похожем или
    # лучшем качестве для табличных признаков (нативная поддержка sklearn,
    # аналог LightGBM) — важно при бюджете в несколько секунд на попытку.
    # ВАЖНО: neg_ratio в build_dataset теперь ближе к реальному дисбалансу
    # (тальк реально занимает ~5-15% кадра), поэтому БЕЗ искусственного
    # sample_weight-балансирования — иначе порог 0.5 на полном снимке
    # оказывается некалиброванным (модель заливает 40-60% кадра, см. историю).
    clf = HistGradientBoostingClassifier(
        max_depth=args.max_depth, max_iter=args.n_estimators,
        learning_rate=0.1, random_state=args.seed,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"[train_talc_segmenter] обучено за {time.time()-t0:.1f}s")

    # Сохраняем СРАЗУ после обучения (до дорогой полной реконструкции масок ниже) —
    # чтобы веса не терялись, если следующий шаг не уложится в лимит по времени.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"model": clf, "feature_names": FEATURE_NAMES,
                     "windows": FEATURE_WINDOWS, "coh_texture_window": COH_TEXTURE_WINDOW}, f)
    print(f"[train_talc_segmenter] модель сохранена: {out_path}")

    if test_samples:
        X_test, y_test, _ = build_dataset(test_samples, rng)
        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"[train_talc_segmenter] held-out (по снимкам) pixel AUC = {auc:.4f}")

        print("[train_talc_segmenter] полная реконструкция маски на held-out снимках "
              "(сравнение с классическим detect_talc по mean_abs_err_pct / IoU):")
        errs, ious = [], []
        for s in test_samples:
            pred_mask, _ = predict_mask(clf, s.bgr, threshold=0.5)
            pred_pct = segment.talc_percentage(pred_mask)
            err = abs(pred_pct - s.gt_pct)
            inter = int(((pred_mask > 0) & (s.gt_mask > 0)).sum())
            union = int(((pred_mask > 0) | (s.gt_mask > 0)).sum())
            iou = inter / union if union else 1.0
            errs.append(err)
            ious.append(iou)
            print(f"    {s.orig_path.name:30s} gt={s.gt_pct:5.2f}% pred={pred_pct:5.2f}% "
                  f"err={err:5.2f} iou={iou:.3f}")
        print(f"[train_talc_segmenter] held-out mean_abs_err_pct={np.mean(errs):.2f}% "
              f"mean_iou={np.mean(ious):.3f}")

    if hasattr(clf, "feature_importances_"):
        importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1])
        print("[train_talc_segmenter] важность признаков:")
        for name, imp in importances:
            print(f"    {name:16s} {imp:.4f}")
    return test_samples


if __name__ == "__main__":
    main()
