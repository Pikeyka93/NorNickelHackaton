# Notebooks — эксперименты, калибровка и отчёт P4

В этой папке лежат только `.ipynb`/`.md`. Данные, веса, отчёты и большие картинки
не коммитятся.

## P4 interface report

`P4_interface_report.ipynb` — сдачный ноутбук по интерфейсу:

- что именно сделано в `app/main.py`;
- как UI связан с `core.analyze()`;
- какие поля показываются геологу;
- какие файлы скачиваются из интерфейса;
- как запустить локально и в режиме заглушки.

## Калибровка детектора талька по синим обводкам

Единственная пиксельная разметка в датасете — синие контуры, нарисованные экспертом
поверх части оталькованных снимков. Их используем как ground truth для талька.

```python
import cv2
from core import segment

bgr = segment.read_image_bgr("<оталькованный снимок с синей обводкой>")
gt_talc = segment.extract_blue_annotations(bgr, fill=True)
pred = segment.analyze_image(bgr)
pred_talc = pred["talc_mask"]

inter = (gt_talc > 0) & (pred_talc > 0)
union = (gt_talc > 0) | (pred_talc > 0)
iou = inter.sum() / max(union.sum(), 1)
err_pct = abs((pred_talc > 0).mean() - (gt_talc > 0).mean()) * 100

print("IoU:", round(iou, 3))
print("talc err %:", round(err_pct, 2))
print("pred talc %:", pred["talc_pct"])
```

Цель для P1: ошибка доли талька около `±3%` на размеченных снимках.

Калибровать в `config.py`:

- `TALC_DARK_PERCENTILE`
- `TALC_ABS_MARGIN`
- `TALC_LOCAL_WINDOW`
- `TALC_LOCAL_MARGIN`
- `TALC_MAX_TEXTURE`
- `TALC_MIN_BLOB_AREA`
- `TALC_BG_WINDOW`
- `TALC_CONTRAST_MARGIN`
- `BLUE_HSV_LOWER / BLUE_HSV_UPPER`, если экспертный синий плохо извлекается.

## Проверка панорам

Панорама — рабочий вход инференса. После калибровки P1 нужно отдельно замерить:

- время обработки одной панорамы;
- корректность тайлинга без швов на маске;
- устойчивость `talc_pct` после даунскейла `MAX_PROCESS_SIDE`.
