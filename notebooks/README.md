# Notebooks — эксперименты и калибровка

Пространство для калибровки порогов. Держите здесь только `.ipynb`/`.md`,
данные и веса не коммитьте (см. `.gitignore`).

## Приоритетная задача: калибровка детектора талька по СИНИМ обводкам

Единственная пиксельная разметка в датасете — синие контуры, нарисованные экспертом
поверх части оталькованных снимков. Используйте их как ground truth:

```python
import cv2
from core import segment
import config as C

bgr = segment.read_image_bgr("<оталькованный снимок с синей обводкой>")
gt_talc = segment.extract_blue_annotations(bgr, fill=True)   # эталон талька (маска)

norm = segment.normalize_illumination(bgr)
sulf = segment.segment_sulfides(norm)
pred_talc = segment.detect_talc(norm, sulf)

# IoU / расхождение доли талька -> подбор порогов в config.py
inter = (gt_talc > 0) & (pred_talc > 0)
union = (gt_talc > 0) | (pred_talc > 0)
iou = inter.sum() / max(union.sum(), 1)
err_pct = abs((pred_talc > 0).mean() - (gt_talc > 0).mean()) * 100
print("IoU", iou, "talc err %", err_pct)   # цель по доле талька: <= ±3%
```

Крутите в `config.py`: `TALC_DARK_PERCENTILE`, `TALC_LOCAL_WINDOW`,
`TALC_MAX_TEXTURE`, `TALC_MIN_BLOB_AREA`, а также `BLUE_HSV_LOWER/UPPER`
(если синий обводки плохо ловятся).

## Калибровка обычные vs тонкие срастания

Соберите несколько эталонных «рядовых» и «труднообогатимых» кропов, посмотрите
распределения `solidity` / `compactness` / `area` по блобам (`classify_intergrowths`),
подберите `INTERGROWTH_*` в `config.py`.

## Порог сульфидов

Если Otsu недооценивает/переоценивает — правьте `SULFIDE_OTSU_OFFSET`.
