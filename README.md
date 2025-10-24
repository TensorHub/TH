# TensorHub

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/main/web/assets/logo.png)

## 📚 О проекте

TensorHub — это мой персональный исследовательский хаб по машинному обучению.

---

## 🧩 Компоненты TensorHub

### 1️⃣ Research
Раздел для чтения и осмысления научных статей.  
Каждая директория — одно исследование с собственным темпом и глубиной.

```
research/
├─ 0001-transformer-math/
│   ├─ theory/
│   ├─ practice/
│   └─ meta.yaml
```

meta.yaml
```
title: "Mathematical Foundations of Transformers"
arxiv_id: "2401.12345"
categories: ["cs.LG", "cs.CL"]
date: "2025-10-22"
status: "in-progress"
tags: ["Transformers", "Optimization", "Theory"]
```

### 2️⃣ ML Radar — Система самооценки

TensorHub содержит встроенный инструмент самодиагностики знаний и навыков.  
Он строит Radar-диаграмму компетенций по 8 осям Machine Learning:

Математика, Данные, Задачи, Парадигмы, Модели, Обучение, Оценка, MLOps

Артефакты хранятся в `assessments/`:

```
assessments/
 ├─ axes.yaml           # структура осей
 ├─ questions/          # банк вопросов
 ├─ responses/          # мои ответы
 ├─ scores/             # рассчитанные баллы
 └─ charts/             # radar-диаграммы
```

Генерация диаграммы выполняется автоматически через GitHub Actions.

---

## 🧮 Как работает самооценка

1. Тесты знаний — 100 вопросов на каждую ось → автоматическая оценка уровня (0–5).
2. Чтение статей — категории arXiv → соответствующие оси (с экспоненциальным затуханием).
3. Практика — результаты экспериментов дают дополнительный вклад в оценку.

Итоговая диаграмма отражает текущий «вектор компетенций» и его динамику по месяцам.

---

## ⚙️ Структура артефактов

```
📦 TensorHub
 ├─ research/        # исследования и эксперименты
 ├─ assessments/     # тесты, результаты, графики
 └─ scripts/         # утилиты для анализа и визуализации
```

---

## 📈 Machine Learning Radar

![ML Radar Chart](assessments/charts/radar_2025-10.png)

Моя цель — достичь экспертного уровня (≈ PhD) по всем восьми направлениям:

| Ось        | Сфера                                                      |
| ---------- | ---------------------------------------------------------- |
| Математика | Линейная алгебра, статистика, оптимизация...               |
| Данные     | Управление данными, приватность, feature engineering...    |
| Задачи     | Постановка, формализация, loss-design...                   |
| Парадигмы  | Supervised, unsupervised, RL, self-supervised...           |
| Модели     | Архитектуры, представления, генеративные модели...         |
| Обучение   | Алгоритмы, regularization, tuning...                       |
| Оценка     | Метрики, интерпретируемость, неопределённость...           |
| MLOps      | CI/CD, мониторинг, reproducibility, fairness...            |

---

## 📓 Принципы

- System over speed — важна глубина, а не частота.
- Reproducibility first — каждый эксперимент должен быть воспроизводим.
- Theory + Practice — знание без применения не имеет веса.
- Continuous self-measurement — регулярная рефлексия и количественная оценка.

---

## 📜 Лицензия

MIT License — см. [LICENSE](LICENSE)

---

<p align="center">
Created as a personal knowledge tensor.  
Learning, one vector at a time.
</p>
