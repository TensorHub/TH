"""
Vision Transformer (ViT) implementation.

Основные компоненты:
- ViTPatchEmbeddings: Извлечение патчей и позиционные эмбеддинги
- ViTMultiHeadAttention: Self-attention механизм
- ViTMLP: Feed-forward сеть
- ViTBlock: Transformer блок (attention + MLP)
- ViT: Полная модель Vision Transformer

Референс: https://arxiv.org/abs/2010.11929 (An Image is Worth 16x16 Words)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ViTPatchEmbeddings(nn.Module):
    """
    Description:
    ---------------
        Извлечение патчей из изображения и добавление позиционных эмбеддингов.

        Vision Transformer обрабатывает изображения как последовательности патчей.
        Этот модуль разбивает изображение на неперекрывающиеся патчи (например, 16×16 пикселей),
        проецирует каждый патч в векторное представление и добавляет позиционную информацию.

        Процесс:
        1. Разбиение изображения на патчи через Conv2d (эквивалентно linear projection)
        2. Flatten и transpose для получения последовательности патчей
        3. Опциональное добавление CLS token в начало (как в BERT)
        4. Добавление позиционных эмбеддингов (суммирование, не конкатенация)

        Математика:
        -----------
        Пусть изображение имеет размер (H, W, C) = (img_size, img_size, 3).
        Количество патчей: N = (H / P)² где P = patch_size

        Извлечение патчей эквивалентно Conv2d:
        - Input: [B, 3, H, W]
        - Conv2d(kernel=P, stride=P): [B, D, H/P, W/P]
        - Reshape: [B, D, N] где N = (H/P)²
        - Transpose: [B, N, D]

        Позиционные эмбеддинги добавляются через суммирование (broadcasting):
        Output = Patches + Position_Embeddings

        Почему суммирование?
        - Экономия параметров (по сравнению с конкатенацией)
        - Сохранение размерности
        - Позволяет модели комбинировать контент и позицию

    Args:
    ---------------
        cfg: VLMConfig с параметрами:
            - vit_img_size (int): Размер входного изображения (предполагается квадратное).
                                  Например: 224, 384, 512.
            - vit_patch_size (int): Размер квадратного патча в пикселях.
                                    Стандартные значения: 16, 32.
                                    img_size должен делиться на patch_size без остатка!
            - vit_hidden_dim (int): Размерность эмбеддинга для каждого патча.
                                    Например: 768 (ViT-Base), 1024 (ViT-Large).
            - vit_cls_flag (bool): Использовать ли CLS token в начале последовательности.
                                   True: как в оригинальном ViT для классификации.
                                   False: используем все патчи (как в CLIP, SigLIP).

    Attributes:
    ---------------
        conv (nn.Conv2d): Сверточный слой для извлечения и проекции патчей.
        cls_token (nn.Parameter, optional): Обучаемый CLS token, добавляемый в начало.
        position_embedding (nn.Parameter): Обучаемые позиционные эмбеддинги для каждой позиции.

    Raises:
    ---------------
        AssertionError: Если img_size не делится на patch_size нацело.

    Examples:
    ---------------
        >>> from models.config import VLMConfig
        >>> import torch

        >>> # Стандартная конфигурация ViT-Base
        >>> cfg = VLMConfig(vit_img_size=224, vit_patch_size=16,
        ...                 vit_hidden_dim=768, vit_cls_flag=False)
        >>> patch_embed = ViTPatchEmbeddings(cfg)
        >>> print(f"Количество патчей: {patch_embed.num_patches}")
        Количество патчей: 196

        >>> # Forward pass
        >>> x = torch.randn(2, 3, 224, 224)  # Батч из 2 изображений
        >>> out = patch_embed(x)
        >>> print(f"Выход: {out.shape}")  # [2, 196, 768]
        Выход: torch.Size([2, 196, 768])

        >>> # С CLS token
        >>> cfg_cls = VLMConfig(vit_img_size=224, vit_patch_size=16,
        ...                     vit_hidden_dim=768, vit_cls_flag=True)
        >>> patch_embed_cls = ViTPatchEmbeddings(cfg_cls)
        >>> out_cls = patch_embed_cls(x)
        >>> print(f"Выход с CLS: {out_cls.shape}")  # [2, 197, 768]
        Выход с CLS: torch.Size([2, 197, 768])

        >>> # Конфигурация nanoVLM (SigLIP)
        >>> cfg_nano = VLMConfig(vit_img_size=512, vit_patch_size=16,
        ...                      vit_hidden_dim=768, vit_cls_flag=False)
        >>> patch_embed_nano = ViTPatchEmbeddings(cfg_nano)
        >>> x_large = torch.randn(1, 3, 512, 512)
        >>> out_large = patch_embed_nano(x_large)
        >>> print(f"nanoVLM выход: {out_large.shape}")  # [1, 1024, 768]
        nanoVLM выход: torch.Size([1, 1024, 768])
    """

    def __init__(self, cfg):
        super().__init__()

        self.img_size    = cfg.vit_img_size
        self.patch_size  = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag    = cfg.vit_cls_flag
        self.embd_dim    = cfg.vit_hidden_dim

        # Проверка корректности размеров
        assert self.img_size % self.patch_size == 0, \
            f"img_size ({self.img_size}) должен делиться на patch_size ({self.patch_size}) без остатка!"

        # ==================== TODO: Conv2d для извлечения патчей ====================
        # Создайте сверточный слой, который разбивает изображение на патчи и проецирует их.
        #
        # Параметры nn.Conv2d:
        #   - in_channels: количество входных каналов (RGB изображение = 3)
        #   - out_channels: размерность эмбеддинга патча (self.embd_dim)
        #   - kernel_size: размер окна свертки (должен быть равен размеру патча)
        #   - stride: шаг свертки (равен kernel_size для неперекрывающихся патчей)
        #   - padding: "valid" означает без паддинга (или можно использовать padding=0)
        #
        # Почему Conv2d эквивалентна извлечению патчей?
        # - kernel_size = patch_size: каждое окно охватывает ровно один патч
        # - stride = patch_size: окна не перекрываются
        # - Веса Conv2d = обучаемая linear projection для каждого патча
        #
        # Пример: img_size=224, patch_size=16
        # Input: [B, 3, 224, 224]
        # Output: [B, embd_dim, 14, 14] где 14 = 224/16

        # TODO: замените на nn.Conv2d(...) с правильными параметрами
        self.conv = None

        # ==================== TODO: Позиционные эмбеддинги ====================
        # Позиционные эмбеддинги позволяют модели "знать", где каждый патч находился на изображении.
        # Без них все патчи выглядят одинаково для attention механизма.
        #
        # Два случая:
        #
        # 1. Если cls_flag=True (как в оригинальном ViT):
        #    - Создайте CLS token: nn.Parameter с формой [1, 1, embd_dim]
        #    - Инициализируйте нулями: torch.zeros(1, 1, self.embd_dim)
        #    - CLS token добавляется в начало последовательности патчей
        #    - Создайте position_embedding: nn.Parameter [1, num_patches + 1, embd_dim]
        #    - +1 потому что CLS token тоже имеет позицию
        #    - Инициализируйте случайными значениями: torch.randn(...)
        #
        # 2. Если cls_flag=False (как в CLIP, SigLIP):
        #    - CLS token не нужен (установите в None)
        #    - Создайте position_embedding: nn.Parameter [1, num_patches, embd_dim]
        #    - Инициализируйте случайными значениями: torch.randn(...)
        #
        # Почему форма [1, num_patches, embd_dim]?
        # - 1: будет broadcast'иться на batch_size при сложении
        # - num_patches: для каждой позиции свой эмбеддинг
        # - embd_dim: той же размерности, что и патчи (для суммирования)

        if self.cls_flag:
            # TODO: Создайте CLS token как Parameter
            self.cls_token = None

            # TODO: Создайте position_embedding для num_patches + 1 позиций
            self.position_embedding = None
        else:
            # TODO: Создайте position_embedding для num_patches позиций
            self.position_embedding = None

        # Вопросы для размышления:
        # - Почему CLS token инициализируется нулями, а position_embedding случайными значениями?
        # - Сколько параметров в Conv2d слое? Формула: kernel_h * kernel_w * in_channels * out_channels + out_channels (bias)
        # - Почему позиционные эмбеддинги обучаемые (Parameter), а не фиксированные?
        # - Что произойдет, если изображение будет другого размера на inference?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Forward pass для извлечения патчей и добавления позиционных эмбеддингов.

            Преобразует изображение в последовательность эмбеддингов патчей с позиционной информацией.
            Выход готов для подачи в Transformer блоки.

        Args:
        ---------------
            x: Входное изображение формы [B, C, H, W] где:
               - B = batch size (количество изображений в батче)
               - C = 3 (каналы RGB)
               - H = img_size (высота)
               - W = img_size (ширина, предполагается квадратное изображение)

        Returns:
        ---------------
            torch.Tensor: Тензор эмбеддингов патчей формы:
                - [B, num_patches, embd_dim] если cls_flag=False
                - [B, num_patches+1, embd_dim] если cls_flag=True

            Каждый элемент последовательности — это эмбеддинг одного патча + позиционная информация.

        Raises:
        ---------------
            RuntimeError: Если размер входного изображения не соответствует ожидаемому img_size.
            AssertionError: Если форма после обработки не соответствует ожидаемой.

        Shape Transformations:
        ----------------------
            Input:  [B, 3, H, W]
            ↓ Conv2d
            [B, D, H/P, W/P]  где D=embd_dim, P=patch_size
            ↓ Flatten(2)
            [B, D, N]  где N=(H/P)*(W/P)=num_patches
            ↓ Transpose(1,2)
            [B, N, D]
            ↓ Add CLS (optional)
            [B, N+1, D] or [B, N, D]
            ↓ Add Position
            Output: [B, N(+1), D]

        Examples:
        ---------------
            >>> cfg = VLMConfig(vit_img_size=224, vit_patch_size=16, vit_hidden_dim=768)
            >>> module = ViTPatchEmbeddings(cfg)
            >>> x = torch.randn(4, 3, 224, 224)
            >>> out = module(x)
            >>> print(out.shape)
            torch.Size([4, 196, 768])
        """

        # Сохраним batch_size для использования в CLS token expansion
        batch_size = x.shape[0]

        # ==================== TODO: Шаг 1 - Извлечение патчей через Conv2d ====================
        # Примените self.conv к входному изображению.
        #
        # Что происходит:
        # - Conv2d с kernel_size=patch_size и stride=patch_size действует как "разрезание" изображения
        # - Каждый патч размером [patch_size, patch_size, 3] проецируется в вектор размерности embd_dim
        # - Веса Conv2d обучаются так, чтобы извлекать полезные признаки из каждого патча
        #
        # Input shape: [B, 3, img_size, img_size]
        # Output shape: [B, embd_dim, img_size/patch_size, img_size/patch_size]
        #
        # Пример: [2, 3, 224, 224] -> [2, 768, 14, 14]

        x = None

        # ==================== TODO: Шаг 2 - Flatten патчей ====================
        # Объедините пространственные размерности (height и width) в одну dimension.
        #
        # Используйте: x.flatten(start_dim=2)
        # - start_dim=2 означает "flatten всё начиная с 3-й размерности"
        # - Размерности 0 (batch) и 1 (channels) остаются нетронутыми
        #
        # Input shape: [B, embd_dim, H', W'] где H'=img_size/patch_size
        # Output shape: [B, embd_dim, H'*W'] = [B, embd_dim, num_patches]
        #
        # Пример: [2, 768, 14, 14] -> [2, 768, 196] где 196 = 14*14

        x = None

        # ==================== TODO: Шаг 3 - Transpose для правильного порядка ====================
        # Поменяйте местами размерности sequence_length и embedding_dim.
        #
        # Используйте: x.transpose(1, 2)
        # - Меняет местами размерности 1 и 2
        #
        # Почему это важно?
        # - Transformer ожидает вход формы [batch, sequence_length, embedding_dim]
        # - После Conv2d мы имеем [batch, embedding_dim, sequence_length]
        # - Нужно поменять местами для правильной работы attention
        #
        # Input shape: [B, embd_dim, num_patches]
        # Output shape: [B, num_patches, embd_dim]
        #
        # Пример: [2, 768, 196] -> [2, 196, 768]

        x = None

        # ==================== TODO: Шаг 4 - Добавление CLS token (если нужен) ====================
        # Если cls_flag=True, добавьте CLS token в начало последовательности.
        #
        # CLS token (Classification Token) - это специальный обучаемый вектор, который:
        # - Добавляется в начало последовательности патчей
        # - Используется для задач классификации (финальное представление изображения)
        # - Взаимодействует со всеми патчами через attention
        #
        # Шаги:
        # 1. Расширить cls_token для текущего batch_size:
        #    self.cls_token имеет форму [1, 1, embd_dim]
        #    Используйте: self.cls_token.expand(batch_size, -1, -1)
        #    Результат: [B, 1, embd_dim]
        #
        # 2. Конкатенировать CLS token с патчами:
        #    Используйте: torch.cat((cls_token, x), dim=1)
        #    - dim=1 означает конкатенация по sequence_length
        #    - CLS token добавляется в начало (перед патчами)
        #
        # Input shape: [B, num_patches, embd_dim]
        # Output shape: [B, num_patches+1, embd_dim]
        #
        # Пример: [2, 196, 768] + [2, 1, 768] -> [2, 197, 768]

        if self.cls_flag:
            # TODO: Расширите cls_token на batch_size
            cls_token = None

            # TODO: Конкатенируйте cls_token с x по dim=1
            x = None

        # ==================== TODO: Шаг 5 - Добавление позиционных эмбеддингов ====================
        # Добавьте позиционную информацию к эмбеддингам патчей.
        #
        # Позиционные эмбеддинги позволяют модели различать патчи по их расположению на изображении.
        # Без них attention не знает, что патч сверху-слева отличается от патча снизу-справа.
        #
        # Суммирование (а не конкатенация):
        # - self.position_embedding имеет форму [1, num_patches(+1), embd_dim]
        # - x имеет форму [B, num_patches(+1), embd_dim]
        # - Broadcasting: [1, N, D] + [B, N, D] = [B, N, D]
        # - Каждому патчу в батче добавляется одинаковый позиционный эмбеддинг
        #
        # Почему суммирование, а не конкатенация?
        # - Экономия памяти и параметров (не увеличиваем размерность)
        # - Модель сама учится комбинировать контент и позицию
        # - Проще для модели работать с фиксированной размерностью
        #
        # Input shape: [B, num_patches(+1), embd_dim]
        # Output shape: [B, num_patches(+1), embd_dim] (форма не меняется)
        #
        # Пример: [2, 196, 768] + [1, 196, 768] -> [2, 196, 768]

        x = None

        return x

        # Вопросы для размышления:
        # - Что произойдет, если забыть сделать transpose на шаге 3?
        # - Почему CLS token добавляется в НАЧАЛО, а не в конец последовательности?
        # - Можно ли использовать фиксированные позиционные эмбеддинги (как sinusoidal в Transformer)?
        # - Как изменится код, если изображение не квадратное (H != W)?
        # - Что случится, если на inference подать изображение другого размера?


# ==================== Placeholder для следующих компонентов ====================
# Эти классы будут реализованы на следующих шагах обучения


class ViTMultiHeadAttention(nn.Module):
    """
    Description:
    ---------------
        Multi-Head Self-Attention механизм для Vision Transformer.
        Позволяет каждому патчу "смотреть" на все остальные патчи и взвешивать их важность.

    Будет реализовано на следующем шаге!
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("Этот компонент будет реализован на следующем шаге обучения")

    def forward(self, x):
        raise NotImplementedError("Этот компонент будет реализован на следующем шаге обучения")


class ViTMLP(nn.Module):
    """
    Description:
    ---------------
        Feed-Forward Network (MLP) для Vision Transformer.
        Применяет нелинейные преобразования к каждому патчу независимо.

    Будет реализовано на следующем шаге!
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("Этот компонент будет реализован на следующем шаге обучения")

    def forward(self, x):
        raise NotImplementedError("Этот компонент будет реализован на следующем шаге обучения")


class ViTBlock(nn.Module):
    """
    Description:
    ---------------
        Transformer блок, объединяющий Attention и MLP с residual connections.
        Базовый строительный блок Vision Transformer.

    Будет реализовано позже!
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("Этот компонент будет реализован позже")

    def forward(self, x):
        raise NotImplementedError("Этот компонент будет реализован позже")


class ViT(nn.Module):
    """
    Description:
    ---------------
        Полная модель Vision Transformer.
        Объединяет patch embeddings и последовательность transformer блоков.

    Будет реализовано в конце!
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("Этот компонент будет реализован в конце")

    def forward(self, x):
        raise NotImplementedError("Этот компонент будет реализован в конце")
