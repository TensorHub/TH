# ML Radar — система самооценки в TensorHub 📊🧠

> «Измеряем прогресс, чтобы осознанно расти». 8 осей ML, детерминированный пайплайн и артефакты как источник истины.

---

## Что это

ML Radar — это внутренняя подсистема TensorHub для регулярной самодиагностики. 
Она агрегирует три сигнала и превращает их в месячную радиальную диаграмму компетенций:

- Тесты знаний: ответы на вопросы по 8 осям (0–5)
- Чтение статей: вклад по arXiv-категориям с экспоненциальным затуханием
- Практика: результаты экспериментов за месяц

Результаты фиксируются в `assessments/scores/` (JSON) и визуализируются в `assessments/charts/` (PNG).

---

## Структура каталога

```
assessments/
├─ axes.yaml                 # Определение 8 осей ML
├─ mappings/
│  └─ arxiv2axis.yaml        # Маппинг arXiv-категорий → оси с весами
├─ questions/                # Банк вопросов (JSONL по оси)
│  ├─ math.jsonl
│  ├─ data.jsonl
│  ├─ tasks.jsonl
│  ├─ paradigms.jsonl
│  ├─ models.jsonl
│  ├─ training.jsonl
│  ├─ evaluation.jsonl
│  └─ mlops.jsonl
├─ monthly/                  # Доп. сигнал «практика» за месяц
│  └─ YYYY-MM/
│     └─ practice.json       # [{ axis, score∈[0,1] }]
├─ responses/                # Ответы тестов по месяцам
│  └─ YYYY-MM.json           # Сгенерировано из веб-формы/Issue
├─ scores/                   # Подсчитанные очки по осям
│  └─ radar_YYYY-MM.json
└─ charts/                   # Диаграммы-радар
   ├─ radar_YYYY-MM.png
   └─ radar_latest.png       # алиас на последнюю диаграмму
```

---

## Оси и маппинги

- `axes.yaml` — декларативный список осей (порядок на диаграмме):

```yaml
Math:
  desc: Линал, статистика, оптимизация
Data:
  desc: Datasets, privacy, feature engineering
Tasks:
  desc: Постановка задач и лоссы
Paradigms:
  desc: Supervised, Self-Supervised, RL
Models:
  desc: Архитектуры и представления
Training:
  desc: Алгоритмы, регуляризация, тюнинг
Evaluation:
  desc: Метрики, интерпретируемость, неопределённость
MLOps:
  desc: CI/CD, мониторинг, воспроизводимость, fairness
```

- `mappings/arxiv2axis.yaml` — вклад категорий arXiv в оси:

```yaml
cs.LG: { Math: 0.3, Models: 0.3, Training: 0.2, Evaluation: 0.2 }
cs.CL: { Paradigms: 0.2, Models: 0.4, Training: 0.2, Evaluation: 0.2 }
cs.CV: { Models: 0.4, Tasks: 0.3, Data: 0.2, Evaluation: 0.1 }
```

---

## Форматы входных данных

- Ответы тестов: `assessments/responses/YYYY-MM.json`

```json
[
  { "axis": "Math", "difficulty": 3, "correct": true },
  { "axis": "Models", "difficulty": 4, "correct": false, "weight": 1.4 }
]
```

- Практика: `assessments/monthly/YYYY-MM/practice.json`

```json
[
  { "axis": "MLOps", "score": 0.8 },
  { "axis": "Evaluation", "score": 0.5 }
]
```

- Метаданные чтения статей берутся из репозитория (категории arXiv в `meta.yaml`) и агрегируются по месяцу с затуханием.

---

## Модель оценивания (коротко и честно)

- Константы (см. `scripts/score_assessment.py`):
  - α = 0.55 — вес тестов
  - β = 0.35 — масштаб чтения
  - γ = 0.10 — масштаб практики
  - Half‑life = 6 месяцев — экспоненциальное затухание вклада чтения

- Итог по оси a:
  - Test(a) ∈ [0,1] → α·5·Test(a)
  - Reading(a) = tanh(β·impact(a))·5
  - Practice(a) = γ·avg(practice(a))·5
  - Score(a) = clamp[0,5]( Test + Reading + Practice )

Все вычисления детерминированы для заданного месяца.

---

## Как пользоваться

### Вариант A — через веб‑форму
1) Откройте `web/assessment.html?m=YYYY-MM`
2) Ответьте на вопросы (демо или полный банк)
3) Сохраните/вставьте JSON в Issue «Assessment: YYYY-MM» (лейбл `assessment`)
4) Дождитесь автосборки — в `assessments/charts/` появится `radar_YYYY-MM.png` и `radar_latest.png`

### Вариант B — вручную в репозитории
1) Создайте `assessments/responses/2025-10.json` (см. формат выше)
2) (Опционально) добавьте `assessments/monthly/2025-10/practice.json`
3) Запустите локально:

```bash
# Зависимости
pip install -U pyyaml matplotlib numpy

# Подсчет очков
python scripts/score_assessment.py \
  --month 2025-10 \
  --repo-root .

# Рендер диаграммы
python scripts/render_radar.py \
  --scores assessments/scores/radar_2025-10.json
```

Артефакты: JSON в `assessments/scores/` и PNG в `assessments/charts/`.

---

## Автоматизация (GitHub Actions)

- Workflow: `.github/workflows/assessment.yml`
  - Триггеры: `schedule` (1‑е число месяца), `push` в `assessments/responses/**` и `assessments/monthly/**`, `workflow_dispatch` с `month`.
  - Шаги: подсчет → рендер → `radar_latest.png` → автокоммит артефактов.

- Напоминание: `.github/workflows/monthly-reminder.yml`
  - Открывает Issue «Assessment: YYYY-MM» со ссылкой на форму.

---

## Хорошие практики

- Держите банк вопросов сбалансированным: 100 вопросов на ось — целевое значение.
- Добавляйте категории arXiv в `research/*/meta.yaml` — это влияет на Reading.
- Фиксируйте практику честно и кратко: 2–5 записей/месяц достаточно.
- Не правьте `scores/` вручную — это артефакты вычислений.

---

## Кастомизация

- Изменить веса и half‑life: правьте константы в `scripts/score_assessment.py` (α, β, γ, HALF_LIFE_MONTHS).
- Переопределить маппинг категорий: `assessments/mappings/arxiv2axis.yaml`.
- Переупорядочить оси и их подписи: `assessments/axes.yaml`.

---

## Траблшутинг

- «Контент не найден» в веб‑модалке: проверьте пути `web/infrastructure/data/index*.json` и наличие `review.md`/`review‑en.md`.
- Диаграмма не обновилась: убедитесь, что workflow завершился успешно и артефакты закоммичены.
- Пустые банки вопросов: веб‑форма перейдёт в демо‑режим (samples). Заполните `questions/*.jsonl`.

---

## Лицензия

MIT. Скрипты и конфигурации свободны для адаптации под личные нужды.

---

## Генерация банка вопросов (LLM‑prompt)

Ниже — универсальный промпт‑шаблон для генерации вопросов в формате JSONL, совместимом с веб‑формой (`web/assessment.html`) и пайплайном. Вставьте значения переменных и передайте промпт модели.

Шаблон переменных:
- `AXIS_NAME` — точное имя оси из `assessments/axes.yaml` (например, «Математика», «Модели», «MLOps»)
- `AXIS_SLUG` — короткий slug (имя файла): `math`, `data`, `tasks`, `paradigms`, `models`, `training`, `evaluation`, `mlops`
- `LANG` — язык вопросов: `ru` или `en`
- `COUNT` — количество вопросов (например, `24` или `100`)
- `START_INDEX` — стартовый индекс для id (например, `1` → `math-0001`)
- `SUBDOMAINS` — список поддоменов (через запятую), берите из `axes.yaml` для консистентности
- `DIFFICULTY_MIX` — распределение сложностей 1..5 в процентах, сумма = 100 (например, `1:10,2:20,3:40,4:20,5:10`)

Как заполнять поля (кратко):
- `AXIS_NAME`: должно в точности совпадать с ключом в `axes.yaml` (язык оси не меняем)
- `AXIS_SLUG`: используется для имени файла и префикса `id` (пример: `math` → `math-0001`)
- `LANG`: влияет на текст (`prompt`, `options`, `explanation`), но не на `axis`
- `COUNT`: выберите 24 для быстрого сэта, 100 — целевой размер банка
- `START_INDEX`: если дополняете существующий файл — поставьте следующий номер (например, 101)
- `SUBDOMAINS`: равномерно покрывайте поддомены (см. `axes.yaml`), например для «Математика»: `ЛинАлг, Оптимизация, Вероятности, Статистика, Инфо-теория`
- `DIFFICULTY_MIX`: стандартный микс — `1:10,2:20,3:40,4:20,5:10`

Схема одной строки JSON (обязательно):
```
{
  "id": "{{AXIS_SLUG}}-NNNN",             // 4 цифры, монотонно от START_INDEX
  "axis": "{{AXIS_NAME}}",                // неизменно как в axes.yaml
  "subdomain": "...",                     // один из SUBDOMAINS
  "difficulty": 1..5,                     // целое, соблюдать DIFFICULTY_MIX
  "type": "mcq"|"truefalse"|"short",
  "prompt": "...",                        // вопрос на LANG
  "options": ["...","...","...","..."],   // только для mcq, ровно 4
  "answer": 0..3 | true|false | "string", // индекс/булево/строка для short
  "explanation": "..."                    // опционально, ≤120 символов
}
```

Быстрый чек‑лист перед сохранением:
- Ровно `COUNT` строк JSON (JSONL), без markdown‑обёрток
- `id` уникальные, монотонные: `{{slug}}-0001`, `{{slug}}-0002`, ...
- `difficulty` ∈ 1..5, примерно соответствует миксу
- `type` корректен; для `mcq` всегда 4 `options`, `answer` в 0..3
- Нет дубликатов `prompt`; дистракторы правдоподобны

Сохранение и использование:
1) Сохраните вывод как `assessments/questions/{{AXIS_SLUG}}.jsonl` (UTF‑8, по одной JSON‑строке на вопрос).
2) Откройте `web/assessment.html` — страница автоматически подхватит файлы из `assessments/questions/` (если нет — возьмёт `samples/`).
3) Сгенерируйте попытку теста, получите JSON ответов и прогоните воркфлоу (см. выше).

Валидация (опционально):
```bash
# Проверка, что каждая строка — валидный JSON
while IFS= read -r line; do echo "$line" | jq -e . >/dev/null || { echo "Invalid JSON"; exit 1; }; done < assessments/questions/{{AXIS_SLUG}}.jsonl
```

Рекомендации по качеству вопросов:
- mcq: один правильный ответ; дистракторы реалистичны и исключают друг друга
- truefalse: формулируйте без двусмысленности
- short: один канонический термин/число, без синонимов
