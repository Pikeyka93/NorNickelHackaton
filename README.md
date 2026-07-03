# Скажи мне, кто твой шлиф

Автоматическая классификация геолого-технологического сорта руды по снимку аншлифа
(отражённая оптическая микроскопия). Кейс Норникеля.

**На выходе:** вердикт о сорте (`ordinary` / `hard_to_process` / `talc`), цветная
маска фаз поверх снимка (🟢 обычные срастания, 🔴 тонкие срастания, 🔵 тальк),
проценты по фазам, текстовое заключение и PDF-отчёт.

## Архитектура — два дополняющих трека

1. **Классификатор сорта** (`core/classifier.py`) — transfer learning
   (EfficientNet-B0 / ResNet50, ImageNet). Классифицирует изображение целиком →
   даёт **вердикт**.
2. **Сегментация фаз** (`core/segment.py`) — классический CV. Даёт **маску и проценты**.

`core/analyze.py` объединяет их: вердикт от классификатора, маска и числа от
сегментации, согласованы через экспертное правило. Если весов классификатора нет —
вердикт считается по процентам (rule-based), система остаётся рабочей.

### Экспертная логика вердикта
- тальк > 10% → `talc`;
- иначе преобладание обычных срастаний → `ordinary`, тонких → `hard_to_process`.

## Структура

```
core/segment.py      # нормализация → сульфиды → срастания → тальк → маска/проценты; тайлинг; синие обводки
core/classifier.py   # обучение и инференс классификатора (сплит по ID шлифа)
core/analyze.py      # контракт analyze() — объединяет оба трека
core/labels.py       # маппинг папок ч1/ч2 в 3 канонических класса + извлечение ID шлифа
core/report.py       # текст-вердикт (RU), CSV, PDF
api/main.py          # FastAPI: upload → analyze() → JSON, batch, CSV, PDF
app/main.py          # Streamlit: загрузка, маска с зумом, метрики, вердикт, экспорт
scripts/train.py     # запуск обучения
config.py            # DATA_DIR, пути к весам, все пороги — в одном месте
Dockerfile           # CUDA 12.2 + проверка GPU
```

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch под CUDA на сервере:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 1) API
uvicorn api.main:app --reload            # http://localhost:8000/docs
# 2) UI
streamlit run app/main.py                # http://localhost:8501
# 3) обучение классификатора
python scripts/train.py --epochs 15
```

Всё работает **сразу**: сегментация — на чистом CV, вердикт — rule-based, пока не
обучен классификатор. `ANALYZE_STUB=1` форсит заглушку (фейковые числа, пустая маска)
для параллельной разработки фронта/бэка.

## Данные

Лежат на сервере, **в git не коммитятся**. Путь по умолчанию (кириллица + пробелы):

```bash
export DATA_DIR="$HOME/dataset/Задача 3. Скажи мне, кто твой шлиф"
python -m core.labels     # проверить маппинг папок и число шлифов по классам
```

Метка класса = имя папки. `ч1` и `ч2` используют разные написания одних и тех же
сортов — `core/labels.py` сводит их в 3 канонических класса
(`тонкие` = `Труднообогатимые руды` = `hard_to_process`).

## Контракт `analyze(image) -> dict`

```python
{
  "image_id": str,
  "verdict": "ordinary" | "hard_to_process" | "talc",
  "metrics": {"sulfide_area_pct", "ordinary_pct", "fine_pct", "talc_pct"},
  "mask_png_b64": str,          # RGBA-маска green/red/blue, прозрачный фон
  "confidence_map_b64": str,    # опц.
  "processing_time_sec": float,
  # + verdict_source, classifier_probs (доп.)
}
```

## Критичные правила (зашиты в код)

- Классификатор: сплит train/val **по ID шлифа** (`GroupShuffleSplit`), не по картинкам.
- Аугментации **обязательно** с `RandomGrayscale` + `ColorJitter` (снимки дико разные
  по цвету — иначе модель учит цвет, а не минералогию).
- Дисбаланс: class weights (обратно пропорц.) + oversampling оталькованных.
- Метрики: macro-F1 + confusion matrix.
- Сегментация: **сначала** per-image нормализация, потом Otsu по нормализованному.
- Панорамы — тайлами с перекрытием, бюджет ≤ 5 мин.
- Тальк калибруется по синим экспертным обводкам (цель по доле ±3%).

## Docker (сервер, L4 / CUDA 12.2)

```bash
docker build -t schlif .
docker run --gpus all -p 8000:8000 -p 8501:8501 schlif   # проверяет, что GPU виден
```

## Разделение работы (4 человека)

- **Классификатор** — `core/classifier.py`, `scripts/train.py`: обучение, подбор
  аугментаций/весов, довести macro-F1.
- **Сегментация** — `core/segment.py` + `notebooks/`: калибровка порогов талька по
  синим обводкам, срастания, тайлинг панорам.
- **API/UI** — `api/main.py`, `app/main.py`: экспорт, batch, режим экспертной проверки.
- **Интеграция/демо** — `core/analyze.py`, `core/report.py`, Docker, отчёт, видео-демо.

Общие точки: контракт `analyze()` и `config.py` — менять согласованно.
