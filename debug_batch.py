import os
import cv2
from pathlib import Path
from core.segment import analyze_image

ORIG_DIR = Path("/home/team077/dataset/Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч1/Оталькованные руды")
EXPERT_DIR = ORIG_DIR / "Области оталькования"
OUT_DIR = Path("debug_batch")

OUT_DIR.mkdir(exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

pairs = []

for orig_path in sorted(ORIG_DIR.iterdir()):
    if not orig_path.is_file():
        continue

    if orig_path.suffix not in exts:
        continue

    expert_path = EXPERT_DIR / orig_path.name

    if expert_path.exists():
        pairs.append((orig_path, expert_path))

print(f"Найдено пар: {len(pairs)}")

for i, (orig_path, expert_path) in enumerate(pairs[:5], start=1):
    name = orig_path.stem

    print(f"\n[{i}/5] {orig_path.name}")

    result = analyze_image(str(orig_path))

    talc_pct = result["talc_pct"]
    print(f"Тальк по алгоритму: {talc_pct:.2f}%")

    original = result["bgr"]
    expert = cv2.imread(str(expert_path))
    mask = result["talc_mask"]
    overlay = result["overlay"]

    cv2.imwrite(str(OUT_DIR / f"{i:02d}_{name}_original.png"), original)
    cv2.imwrite(str(OUT_DIR / f"{i:02d}_{name}_expert.png"), expert)
    cv2.imwrite(str(OUT_DIR / f"{i:02d}_{name}_mask.png"), mask)
    cv2.imwrite(str(OUT_DIR / f"{i:02d}_{name}_overlay.png"), overlay)

print("\nГотово. Файлы сохранены в папку debug_batch/")
