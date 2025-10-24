# Вопросы для ежемесячной оценки

- Формат: по оси 100 вопросов в JSONL.
- Поля: `id`, `axis`, `subdomain`, `difficulty (1-5)`, `type` (mcq/truefalse/text), `prompt`, `options` (если mcq), `answer`, `explanation`.
- Пример файлов: см. `samples/`.

Структура:
- Один файл на ось: `math.jsonl`, `data.jsonl`, `tasks.jsonl`, `paradigms.jsonl`, `models.jsonl`, `training.jsonl`, `evaluation.jsonl`, `mlops.jsonl`.
