# TASKS — зоны ответственности и чек-лист

Скоуп по уточнениям жюри: **сегментируем только тальк**; сорт руды
(`ordinary` / `hard_to_process` / `talc`) определяет **только классификатор**;
никакой морфологии/подсчёта срастаний. Панорама — рабочий вход инференса (тайлинг).
**Деплой (Streamlit + Cloudflare) ломать нельзя** — без живой ссылки риск 0 баллов.

## Владельцы
- **P1** — сегментация талька
- **P2** — классификатор сорта + обучение
- **P3** — API, Docker, зависимости, стабильность деплоя
- **P4** — UI, отчёты, обработка ошибок, оформление демо

## Чек-лист

| Раздел | Файл | Владелец | Что сделать | Статус |
|---|---|---|---|---|
| Сегментация талька | `core/segment.py` | P1 | нормализация + детектор талька (тёмная гладкая фаза, темнее локального фона), маска, `talc_pct`, тайлинг | ✅ каркас готов |
| Калибровка талька | `core/segment.py`, `config.py`, `notebooks/` | P1 | подобрать `TALC_*` по синим обводкам (`extract_blue_annotations`), цель ошибка `talc_pct` ±3% | ⬜ TODO |
| Тайлинг панорам | `core/segment.py` | P1 | проверить бюджет ≤5 мин на реальной панораме; при нужде — ленивый тайлинг (pyvips/tifffile) | ⬜ TODO |
| Классификатор | `core/classifier.py` | P2 | 3 класса, EfficientNet-B0/ResNet50, сплит по ID шлифа, grayscale+colorjitter, class weights + oversampling | ✅ код готов |
| Обучение | `scripts/train.py` | P2 | обучить на `DATA_DIR`, сохранить веса, macro-F1 + confusion matrix | ⬜ TODO |
| Маппинг папок | `core/labels.py` | P2 | сверить имена папок ч1/ч2 (`python -m core.labels`) | ⬜ проверить |
| Контракт | `core/analyze.py` | P4 | `verdict`+`talc_pct`+`talc_mask_png_b64`+`consistency_check`+`classifier_confidence` | ✅ готов |
| API | `api/main.py` | P3 | `/analyze`, `/analyze/batch`, `/export/csv`, `/report/pdf`, логи | ✅ готов |
| Зависимости | `requirements.txt` | P3 | torch/torchvision зафиксированы под cu121; `--no-cache-dir` | ✅ готов |
| Docker | `Dockerfile` | P3 | CUDA 12.2, проверка GPU; не ломать деплой | ✅ готов |
| UI | `app/main.py` | P4 | вердикт, синяя маска талька (зум), `talc_pct`, `consistency_check`, экспорт; `analyze()` в try/except | ✅ готов |
| Отчёты | `core/report.py` | P4 | CSV + PDF (тальк-only), Cyrillic-шрифт | ✅ готов |
| Демо/сдача | — | P4 | видео ≤5 мин, презентация, живая ссылка, архив кода | ⬜ TODO |

## Как запустить
```bash
export DATA_DIR="$HOME/dataset/Задача 3. Скажи мне, кто твой шлиф"
python -m core.labels                      # проверить маппинг папок
python scripts/train.py --epochs 15        # P2: обучение (нужен GPU)
uvicorn api.main:app --host 0.0.0.0 --port 8000   # P3: API
streamlit run app/main.py                  # P4: UI (живой деплой)
```
Пока весов классификатора нет — вердикт fallback (talc>10% → talc, иначе ordinary),
сегментация талька работает сразу. `ANALYZE_STUB=1` — заглушка для разработки фронта.
