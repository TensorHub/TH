# 🎓 DeepEncoder Learning Project

> Учебное воссоздание архитектуры DeepSeek-OCR с нуля для глубокого понимания Vision Transformers

---

## 🎯 Цель проекта

**Образовательная миссия:** Понять каждую строку кода, каждое архитектурное решение и каждый trade-off в DeepEncoder через практическую реализацию.

**Что вы получите:**
- ✓ Глубокое понимание Vision Transformers (от базовых концепций до продвинутых техник)
- ✓ Опыт работы с Attention mechanisms (Global, Window, Relative Positions)
- ✓ Знание техник компрессии визуальных токенов
- ✓ Портфолио: рабочая реализация + CookBook с визуализациями

---

## 🏗️ Архитектура DeepEncoder

```
┌────────────────────────────────────────────────────────────────┐
│                   DeepEncoder (380M параметров)                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Image (B, 3, 1024, 1024)                               │
│          │                                                      │
│          ├──────────────────────────────────────┐              │
│          │                                       │              │
│   ┌──────▼────────┐                    ┌────────▼──────┐      │
│   │  SAM-base     │                    │  CLIP-large   │      │
│   │  80M params   │                    │  300M params  │      │
│   │  ───────────  │                    │  ───────────  │      │
│   │ Window Attn   │                    │ Global Attn   │      │
│   │ 14×14 windows │                    │ Full context  │      │
│   │ Relative Pos  │                    │ Absolute Pos  │      │
│   └───────┬───────┘                    └───────┬───────┘      │
│           │                                    │              │
│   ┌───────▼────────┐                          │              │
│   │ 16x Compressor │                          │              │
│   │ 2-layer Conv   │                          │              │
│   │ 4096 → 256     │                          │              │
│   └───────┬────────┘                          │              │
│           │                                    │              │
│           └────────────┬───────────────────────┘              │
│                        │                                      │
│               ┌────────▼──────────┐                           │
│               │  Fused Features   │                           │
│               │  (B, 513, D)      │                           │
│               └───────────────────┘                           │
│                        │                                      │
│               ┌────────▼──────────┐                           │
│               │   LLM Decoder     │  (вне проекта)           │
│               │  (OCR generation) │                           │
│               └───────────────────┘                           │
└────────────────────────────────────────────────────────────────┘
```

**Философия:** Local Processing → Compression → Global Understanding

---

## 📂 Структура проекта

```
DeepEncoder/
├── README.md                    # 👈 Вы здесь
├── src/                         # Исходный код (создаём по этапам)
│   ├── embeddings/              # Этап 1: PatchEmbedding
│   ├── positional/              # Этап 2: Absolute/Relative Pos
│   ├── projectors/              # Этап 3: MLP Projectors
│   ├── attention/               # Этап 4: SDPA, Flash, Window
│   ├── blocks/                  # Этап 5: Transformer Blocks
│   ├── models/                  # Этап 6-7: CLIP, SAM
│   └── compression/             # Этап 8: Integration
├── tests/                       # Unit-тесты (TDD подход)
├── configs/                     # Конфигурации моделей
└── notebooks/                   # CookBook (финальный этап)

Референсная реализация:
../DeepSeek-OCR/deepencoder/     # Официальный код (1058 строк)
├── build_linear.py              # MlpProjector (174 строки)
├── clip_sdpa.py                 # CLIP-large ViT (354 строки)
└── sam_vary_sdpa.py             # SAM-base ViT (529 строк)
```

---

## 🗺️ RoadMap: 8 этапов обучения

> **Детальный план:** См. `../../memory/memory-bank/roadmap.md`

### От простого к сложному (6-8 недель)

| Этап | Название | Сложность | Время | Статус |
|------|----------|-----------|-------|--------|
| 0 | Подготовка и настройка | ⚙️ | 1 день | ⏳ 80% |
| 1 | **PatchEmbedding** | ★☆☆☆☆ | 2-3 дня | ⏸️ Следующий |
| 2 | Positional Encodings | ★★☆☆☆ | 3-4 дня | ⏸️ |
| 3 | MLP Projectors | ★★★☆☆ | 3-4 дня | ⏸️ |
| 4 | Attention Mechanisms | ★★★★☆ | 5-7 дней | ⏸️ |
| 5 | Transformer Blocks | ★★★☆☆ | 2-3 дня | ⏸️ |
| 6 | CLIP-large ViT | ★★★★☆ | 4-5 дней | ⏸️ |
| 7 | SAM-base ViT | ★★★★★ | 5-7 дней | ⏸️ |
| 8 | DeepEncoder + CookBook | ★★★★★ | 7-10 дней | ⏸️ |

---

## 🚀 Быстрый старт

### Этап 0: Подготовка (завершите перед началом кодирования)

#### 1. Проверьте окружение

```bash
# Python и PyTorch
python --version  # Должно быть 3.10+
python -c "import torch; print(torch.__version__)"  # Должно быть 2.0+
python -c "import torch; print(torch.cuda.is_available())"  # True (или False если CPU)

# Установите зависимости
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install easydict pytest

# Flash Attention (опционально, можно использовать SDPA)
# pip install flash-attn --no-build-isolation
```

#### 2. Изучите Банк Памяти

📚 **Обязательное чтение перед началом:**

- `../../memory/memory-bank/projectbrief.md` — Цели проекта и Definition of Done
- `../../memory/memory-bank/roadmap.md` — Детальный план 8 этапов
- `../../memory/memory-bank/techContext.md` — Техническое описание компонентов
- `../../memory/memory-bank/systemPatterns.md` — 8 ключевых архитектурных паттернов

#### 3. Изучите референсную реализацию

```bash
# Официальный код для референса
ls -lh ../DeepSeek-OCR/deepencoder/

# Прочитайте файлы в этом порядке:
# 1. build_linear.py (проекторы — самый простой)
# 2. clip_sdpa.py (CLIP — знакомая архитектура)
# 3. sam_vary_sdpa.py (SAM — самый сложный)
```

---

## 📖 Методология обучения: Socratic Method + TDD

### 🎓 Образовательная стратегия

**Роль наставника:**
- Задаёт наводящие вопросы вместо готовых решений
- Даёт минимальные подсказки (≤8 строк кода)
- Предлагает планы и критерии проверки
- Поощряет Test-Driven Development

**Роль ученика (вы):**
- Самостоятельно пишете код (основную часть)
- Отвечаете на вопросы письменно (фиксация понимания)
- Рисуете схемы и диаграммы
- Запускаете тесты и анализируете ошибки

### 🧪 Test-Driven Development (TDD)

**Цикл разработки:**
```
1. Напишите тест (который падает)
   ↓
2. Реализуйте минимум кода для прохождения теста
   ↓
3. Запустите тест (должен пройти)
   ↓
4. Рефакторинг (улучшите код)
   ↓
5. Повторите для следующей функции
```

**Пример для Этапа 1:**
```python
# tests/test_patch_embed.py
def test_patch_embed_shapes():
    """Проверка корректности shapes."""
    patch_embed = PatchEmbed(patch_size=16, embed_dim=768)
    x = torch.randn(2, 3, 256, 256)
    out = patch_embed(x)

    expected_patches = (256 // 16) ** 2  # 256
    assert out.shape == (2, 16, 16, 768)

# Сначала тест падает → потом реализуем PatchEmbed → тест проходит ✓
```

---

## 🧭 Начало Этапа 1: PatchEmbedding

> **Детали:** См. `../../memory/memory-bank/roadmap.md` → Этап 1

### Теория: От NLP к Vision

**В NLP:**
```
"Hello world" → Tokenizer → [101, 7592, 2088, 102]
```

**В Vision:**
```
Image (3, 224, 224) → Conv2d(kernel=14, stride=14) → Patches (256, 768)
                      ↓
         16×16 патчей размером 14×14 пикселей каждый
```

### 3 ключевых вопроса для исследования

Прежде чем начать код, ответьте на эти вопросы (письменно):

1. **Почему используется Conv2d вместо reshape?**
   - Подсказка: Learnable transformation vs fixed operation

2. **Зачем нужен CLS токен?**
   - Подсказка: Аналогия с BERT `[CLS]` для классификации

3. **Как меняется число патчей при изменении разрешения?**
   - Формула: `num_patches = (H / patch_size) × (W / patch_size)`

### Задачи реализации

**1.1 Простая патчификация (SAM-style)**
```python
# src/embeddings/patch_embed.py

class PatchEmbed(nn.Module):
    """
    Преобразует изображение в последовательность патч-эмбеддингов.

    TODO: Реализовать Conv2d патчификацию
    TODO: Объяснить permute: (B, C, H, W) → (B, H, W, C)
    """
    pass
```

**1.2 CLIP PatchEmbedding с CLS токеном**
```python
# src/embeddings/clip_embeddings.py

class CLIPVisionEmbeddings(nn.Module):
    """
    CLIP-style: PatchEmbedding + CLS token + Positional Embeddings.

    TODO: Реализовать patch_embedding (Conv2d)
    TODO: Создать learnable class_embedding (Parameter)
    TODO: Конкатенация CLS: torch.cat([cls, patches], dim=1)
    """
    pass
```

### Критерии завершения Этапа 1

- ✓ Реализованы `PatchEmbed` и `CLIPVisionEmbeddings`
- ✓ Тесты проходят для разных разрешений (224, 256, 512)
- ✓ Написаны docstrings с аналогиями к NLP
- ✓ Вы можете объяснить trade-off: Conv2d vs Linear

---

## 📚 Полезные ресурсы

### Документация проекта

- **Банк Памяти:** `../../memory/memory-bank/`
  - `projectbrief.md` — Цели и DoD
  - `roadmap.md` — 8 этапов (детально)
  - `techContext.md` — Технические детали
  - `systemPatterns.md` — Архитектурные паттерны
  - `activeContext.md` — Текущий фокус
  - `progress.md` — Трекинг прогресса
  - `../rules/memory-bank.mdc` — Журнал обучения

### Референсная реализация

- `../DeepSeek-OCR/deepencoder/` — Официальный код (1058 строк)
- `../../paper/DeepSeek-OCR.pdf` — Научная статья
- `../../review/review.md` — Обзор и мотивация

### Внешние ресурсы

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Базовая статья по Transformers
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — Vision Transformer (ViT)
- [CLIP Paper](https://arxiv.org/abs/2103.00020) — Contrastive Language-Image Pre-training
- [SAM Paper](https://arxiv.org/abs/2304.02643) — Segment Anything Model

---

## 📊 Definition of Done (DoD)

Проект считается завершённым, когда вы можете:

- ✓ **Понимание:** Объяснить каждую строку кода и архитектурное решение
- ✓ **Forward pass:** Запустить forward pass для всех компонентов
- ✓ **Тесты:** Unit-тесты покрывают все компоненты
- ✓ **Trade-offs:** Объяснить компромиссы каждого решения
- ✓ **CookBook:** Jupyter notebook с визуализациями и экспериментами

---

## 🤝 Как получить помощь

### Формат взаимодействия

1. **Прочитайте теорию** в RoadMap для текущего этапа
2. **Ответьте на вопросы** из этапа (письменно)
3. **Попробуйте реализовать** самостоятельно
4. **Запустите тесты** и проанализируйте ошибки
5. **Если застряли:** Задайте конкретный вопрос с контекстом

**Хороший вопрос:**
> "Я реализовал PatchEmbed, но при разрешении 256×256 получаю shape (2, 256, 16, 16, 768)
> вместо ожидаемого (2, 16, 16, 768). Вот мой код: [код].
> Где я ошибся в понимании reshape операции?"

**Плохой вопрос:**
> "Как реализовать PatchEmbed?"

---

## 📝 Следующие шаги

1. ✅ Прочитайте этот README полностью
2. ✅ Изучите `../../memory/memory-bank/roadmap.md` (Этап 1)
3. ✅ Проверьте окружение (Python, PyTorch, CUDA)
4. ✅ Прочитайте референс: `../DeepSeek-OCR/deepencoder/clip_sdpa.py:107-156`
5. ✅ Ответьте на 3 вопроса из "Начало Этапа 1"
6. ✅ Создайте `src/embeddings/` и начните с тестов

---

## 🎯 Мотивация

> "Я слышу и забываю. Я вижу и запоминаю. Я делаю и понимаю." — Конфуций

Этот проект — не просто копирование кода. Это путешествие от незнания к глубокому пониманию Vision Transformers через практику. Каждая строка кода, которую вы напишете, каждый тест, который вы пройдёте, каждый trade-off, который вы поймёте — это шаг к мастерству.

**Удачи в обучении!** 🚀

---

**Версия:** 1.0.0
**Последнее обновление:** 2025-10-31
**Автор:** DeepSeek-OCR Learning Project
