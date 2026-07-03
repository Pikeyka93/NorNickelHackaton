# ============================================================
# ВЛАДЕЛЕЦ: P2 (используется классификатором) — маппинг папок + ID шлифа
# TODO(P2): сверить имена папок ч1/ч2 на реальном DATA_DIR (python -m core.labels).
# ============================================================
"""
Folder -> canonical class mapping and slide-ID extraction.

There is NO machine-readable annotation — the label is the folder name. Part 1 and
Part 2 use different spellings for the same three sorts, so we normalise both to the
3 canonical classes from config.

CRITICAL: the classifier must split train/val by SLIDE ID (not by image), because one
slide produces several field-of-view photos ("2652976 10x 2", "2652976 10x 3", ...).
`extract_slide_id` pulls that group key so GroupShuffleSplit can keep a slide entirely
in train or entirely in val (no leakage, honest F1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from config import (
    CLASS_HARD,
    CLASS_ORDINARY,
    CLASS_TALC,
    DATA_DIR,
    PANORAMA_DIR,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Explicit spellings we know about (ч1 and ч2). Lower-cased, stripped.
_EXPLICIT = {
    # part 1
    "рядовые руды": CLASS_ORDINARY,
    "труднообогатимые руды": CLASS_HARD,
    "оталькованные руды": CLASS_TALC,
    # part 2
    "рядовые": CLASS_ORDINARY,
    "тонкие": CLASS_HARD,
    "оталькованные": CLASS_TALC,
}


def canonical_class_for_folder(folder_name: str) -> Optional[str]:
    """Map a single folder name to a canonical class, or None if it isn't a class folder.

    Falls back to keyword matching so minor spelling/case variants still resolve
    (e.g. 'Тонкие срастания' -> hard_to_process).
    """
    key = folder_name.strip().lower()
    if key in _EXPLICIT:
        return _EXPLICIT[key]

    # Keyword fallback. Order matters: talc first (most specific), then hard, then ordinary.
    if "тальк" in key or "оталь" in key:
        return CLASS_TALC
    if "трудно" in key or "тонк" in key:
        return CLASS_HARD
    if "рядов" in key:
        return CLASS_ORDINARY
    return None


def canonical_class_for_path(path: Path, data_dir: Path = DATA_DIR) -> Optional[str]:
    """Walk a file's parent folders (nearest first) and return the first class match."""
    try:
        rel = path.resolve().relative_to(Path(data_dir).resolve())
        parts = list(rel.parts[:-1])  # drop the filename
    except (ValueError, OSError):
        parts = [p.name for p in path.parents]
    for name in reversed(parts):  # nearest parent first
        cls = canonical_class_for_folder(name)
        if cls is not None:
            return cls
    return None


# Slide ID = leading run of >=4 digits in the filename (e.g. "2652976 10x 2" -> "2652976").
_SLIDE_ID_RE = re.compile(r"(\d{4,})")


def extract_slide_id(filename: str) -> str:
    """Group key for GroupShuffleSplit. Uses the first long digit run; falls back to
    a magnification/index-stripped stem so unusual names still group sensibly."""
    stem = Path(filename).stem
    m = _SLIDE_ID_RE.search(stem)
    if m:
        return m.group(1)
    # Fallback: strip trailing magnification ("10x") and index tokens, keep the rest.
    cleaned = re.sub(r"\b\d+\s*[xх]\b", "", stem, flags=re.IGNORECASE)  # latin x and cyrillic х
    cleaned = re.sub(r"[\s_]+\d+$", "", cleaned).strip()
    return cleaned or stem


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str        # canonical class
    slide_id: str     # group key


def iter_samples(data_dir: Path = DATA_DIR, include_panoramas: bool = False) -> Iterator[Sample]:
    """Yield every labelled image under data_dir as (path, label, slide_id).

    Panoramas are skipped by default (they're huge stitches, not per-class training
    crops); flip include_panoramas to pull them in for inference-time testing.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"DATA_DIR not found: {data_dir}\n"
            "Set the DATA_DIR env var to the dataset root (see config.py)."
        )
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if not include_panoramas and PANORAMA_DIR.lower() in str(p).lower():
            continue
        label = canonical_class_for_path(p, data_dir)
        if label is None:
            continue
        yield Sample(path=p, label=label, slide_id=extract_slide_id(p.name))


def load_dataset_index(data_dir: Path = DATA_DIR) -> list[Sample]:
    """Materialise the sample list and print a quick class/slide summary."""
    samples = list(iter_samples(data_dir))
    from collections import Counter

    by_class = Counter(s.label for s in samples)
    slides_by_class = {
        c: len({s.slide_id for s in samples if s.label == c}) for c in by_class
    }
    print(f"[labels] {len(samples)} images from {data_dir}")
    for c in sorted(by_class):
        print(f"[labels]   {c:16} images={by_class[c]:5d}  slides={slides_by_class[c]}")
    return samples


if __name__ == "__main__":
    # Quick sanity run: python -m core.labels
    load_dataset_index()
