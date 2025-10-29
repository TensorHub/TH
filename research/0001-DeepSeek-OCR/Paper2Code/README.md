# nanoVLM Educational Reproduction

> 🎓 Учебный проект по воспроизведению Vision-Language Model в режиме Socratic tutoring

## 📚 О проекте

Это образовательная реализация Vision-Language Model (VLM) на основе [nanoVLM](https://github.com/huggingface/nanoVLM) от HuggingFace. Проект создан для **глубокого понимания** архитектуры VLM через пошаговую реализацию каждого компонента.

### Чему вы научитесь

- ✅ Как работает Vision Transformer (ViT)
- ✅ Архитектура современных Language Models (GQA, RoPE)
- ✅ Как объединить визуальные и языковые модальности
- ✅ Процесс обучения multimodal моделей
- ✅ Эффективная генерация с KV-cache

### Отличия от оригинального nanoVLM

| Аспект | nanoVLM | Этот проект |
|--------|---------|-------------|
| Формат | Готовый код | TODO с наводящими вопросами |
| Подход | Reference implementation | Socratic tutoring |
| Документация | Комментарии | Подробные докстринги + примеры |
| Цель | Рабочая модель | Понимание архитектуры |

## 🏗️ Структура проекта

```
Paper2Code/
├── src/
│   ├── models/           # Компоненты модели
│   │   ├── config.py              # Конфигурация
│   │   ├── vision_transformer.py  # ViT (ТЕКУЩИЙ ФОКУС)
│   │   ├── language_model.py      # Language Model (скоро)
│   │   ├── modality_projector.py  # Projector (скоро)
│   │   └── vision_language_model.py  # VLM (скоро)
│   ├── data/             # Data pipeline
│   └── utils/            # Вспомогательные функции
├── tests/                # Unit тесты
├── memory/               # Банк Памяти (документация)
└── README.md            # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Минимальные зависимости
pip install torch numpy

# Для тестов
pip install pytest

# Опционально (для обучения)
pip install torchvision pillow datasets transformers wandb
```

### 2. Текущая задача: Реализация ViTPatchEmbeddings

**Файл**: `src/models/vision_transformer.py`

**Что нужно сделать**:
1. Открыть файл `src/models/vision_transformer.py`
2. Найти класс `ViTPatchEmbeddings`
3. Реализовать TODO в методах `__init__` и `forward`
4. Запустить тесты для проверки: `pytest tests/test_vision_transformer.py -v`

### 3. Проверка реализации

```bash
# Запустить все тесты
pytest tests/ -v

# Запустить только тесты для ViTPatchEmbeddings
pytest tests/test_vision_transformer.py::TestViTPatchEmbeddings -v
```

## 📖 Учебный процесс

### Формат обучения

Мы используем **Socratic tutoring** подход:

1. **TODO с подсказками** - вместо готового кода
2. **Наводящие вопросы** - для развития мышления
3. **Примеры и докстринги** - для понимания контекста
4. **Тесты** - для проверки правильности

### Пример TODO

```python
def forward(self, x):
    """Подробный докстринг с примерами..."""

    # TODO: Шаг 1 - Применить Conv2d для извлечения патчей
    # Используйте self.conv(x)
    x = None  # TODO: замените

    # TODO: Шаг 2 - Flatten патчи в одну размерность
    # Используйте x.flatten(2)
    x = None  # TODO: замените

    # Вопросы для размышления:
    # - Почему Conv2d эквивалентна извлечению патчей?
    # - Зачем нужен transpose после flatten?

    return x
```

### Уровни помощи

Если застряли, можно получить подсказки разного уровня:

- **Level 0**: Прочитать докстринг и комментарии TODO
- **Level 1**: Задать уточняющие вопросы
- **Level 2**: Получить подсказку о методе/функции
- **Level 3**: Увидеть частичную реализацию
- **Level 4**: Получить полное решение (только при явном запросе!)

## 🎯 Roadmap

### ✅ Milestone 1: Инициализация (ЗАВЕРШЕНО)
- [x] Анализ nanoVLM
- [x] Создание Банка Памяти
- [x] Структура проекта

### 🔄 Milestone 2: Vision Transformer (В РАБОТЕ)
- [x] Структура проекта создана
- [x] Конфигурация (config.py)
- [ ] **ViTPatchEmbeddings (ТЕКУЩАЯ ЗАДАЧА)**
- [ ] ViTMultiHeadAttention
- [ ] ViTMLP
- [ ] ViTBlock
- [ ] ViT (полная модель)

### ⏳ Milestone 3: Language Model
- [ ] RMSNorm
- [ ] RotaryEmbedding
- [ ] LanguageModelGroupedQueryAttention
- [ ] LanguageModelMLP
- [ ] LanguageModelBlock
- [ ] LanguageModel

### ⏳ Milestone 4-10: Далее...
См. `memory/memory-bank/progress.md` для полного roadmap.

## 📚 Ресурсы

### Документация проекта
- `memory/memory-bank/projectbrief.md` - Цели и требования
- `memory/memory-bank/techContext.md` - Технологии и архитектура
- `memory/memory-bank/systemPatterns.md` - Паттерны проектирования
- `memory/memory-bank/progress.md` - Детальный трекинг прогресса

### Референсная реализация
- Оригинальный nanoVLM: `/Users/me/Documents/nanoVLM`
- GitHub: https://github.com/huggingface/nanoVLM

### Статьи
- Vision Transformer: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- SigLIP: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- GQA: [Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)

## 🤝 Как работать с проектом

### Основной workflow

1. **Прочитать задачу** в текущем файле (сейчас: `vision_transformer.py`)
2. **Понять математику** из докстринга и комментариев
3. **Реализовать TODO** пошагово
4. **Запустить тесты** для проверки
5. **Задать вопросы** если что-то неясно
6. **Перейти к следующему компоненту**

### Контрольные вопросы (после ViTPatchEmbeddings)

После реализации компонента, попробуйте ответить:

1. Почему Conv2d с stride=patch_size эквивалентна извлечению патчей?
2. Сколько параметров в Conv2d слое? Как это посчитать?
3. Зачем нужны позиционные эмбеддинги в Vision Transformer?
4. В чем разница между суммированием и конкатенацией позиционных эмбеддингов?
5. Что произойдет, если забыть добавить позиционные эмбеддинги?

## 🐛 Отладка

### Типичные ошибки

1. **Shape mismatch**:
   ```python
   # Добавьте проверки размерностей
   print(f"x.shape после Conv2d: {x.shape}")
   assert x.shape == expected_shape, f"Expected {expected_shape}, got {x.shape}"
   ```

2. **Тесты не проходят**:
   - Проверьте, что все TODO заменены на код
   - Убедитесь, что размерности соответствуют ожидаемым
   - Посмотрите на референсную реализацию (но только после попытки!)

3. **Не понимаю математику**:
   - Задайте вопрос с конкретным примером
   - Попробуйте на маленьких числах (2x2 матрица)
   - Прочитайте докстринг еще раз внимательно

## 📝 Лицензия

MIT License - используйте свободно для обучения и экспериментов.

## 🙏 Acknowledgments

- **nanoVLM** от HuggingFace - референсная реализация
- **nanoGPT** от Andrej Karpathy - вдохновение для минималистичного подхода
- **Socratic Method** - педагогический подход

---

**Готовы начать? Откройте `src/models/vision_transformer.py` и начните с `ViTPatchEmbeddings`! 🚀**
