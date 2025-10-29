"""
Unit tests for Vision Transformer components.

Запуск:
    pytest tests/test_vision_transformer.py -v
    или
    python -m pytest tests/test_vision_transformer.py -v
"""

import torch
import pytest
import sys
from pathlib import Path

# Добавляем src в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.config import VLMConfig
from models.vision_transformer import ViTPatchEmbeddings


class TestViTPatchEmbeddings:
    """Тесты для ViTPatchEmbeddings."""

    @pytest.fixture
    def cfg(self):
        """Базовая конфигурация для тестов."""
        return VLMConfig(
            vit_img_size=224,
            vit_patch_size=16,
            vit_hidden_dim=768,
            vit_cls_flag=False
        )

    @pytest.fixture
    def cfg_with_cls(self):
        """Конфигурация с CLS token."""
        return VLMConfig(
            vit_img_size=224,
            vit_patch_size=16,
            vit_hidden_dim=768,
            vit_cls_flag=True
        )

    def test_initialization(self, cfg):
        """Тест инициализации модуля."""
        module = ViTPatchEmbeddings(cfg)

        # Проверяем, что атрибуты установлены корректно
        assert module.img_size == 224
        assert module.patch_size == 16
        assert module.num_patches == 196  # (224/16)^2
        assert module.embd_dim == 768
        assert module.cls_flag == False

        # Проверяем, что Conv2d создан с правильными параметрами
        assert module.conv is not None, "Conv2d слой не создан! Проверьте TODO в __init__"
        assert isinstance(module.conv, torch.nn.Conv2d)
        assert module.conv.in_channels == 3
        assert module.conv.out_channels == 768
        assert module.conv.kernel_size == (16, 16)
        assert module.conv.stride == (16, 16)

        # Проверяем позиционные эмбеддинги
        assert module.position_embedding is not None, "position_embedding не создан! Проверьте TODO в __init__"
        assert module.position_embedding.shape == (1, 196, 768)

    def test_initialization_with_cls(self, cfg_with_cls):
        """Тест инициализации с CLS token."""
        module = ViTPatchEmbeddings(cfg_with_cls)

        assert module.cls_flag == True

        # Проверяем CLS token
        assert module.cls_token is not None, "CLS token не создан! Проверьте TODO в __init__"
        assert module.cls_token.shape == (1, 1, 768)

        # Проверяем позиционные эмбеддинги (должны включать позицию для CLS)
        assert module.position_embedding.shape == (1, 197, 768)  # 196 + 1

    def test_forward_shape_without_cls(self, cfg):
        """Тест размерности выхода без CLS token."""
        module = ViTPatchEmbeddings(cfg)

        # Создаем фейковый батч изображений
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        out = module(x)

        # Проверяем размерность
        expected_shape = (batch_size, 196, 768)  # [B, num_patches, hidden_dim]
        assert out.shape == expected_shape, \
            f"Неправильная размерность! Ожидается {expected_shape}, получено {out.shape}. Проверьте forward pass."

    def test_forward_shape_with_cls(self, cfg_with_cls):
        """Тест размерности выхода с CLS token."""
        module = ViTPatchEmbeddings(cfg_with_cls)

        batch_size = 3
        x = torch.randn(batch_size, 3, 224, 224)

        out = module(x)

        # Размерность должна включать CLS token
        expected_shape = (batch_size, 197, 768)  # [B, num_patches + 1, hidden_dim]
        assert out.shape == expected_shape, \
            f"Неправильная размерность с CLS! Ожидается {expected_shape}, получено {out.shape}"

    def test_different_img_sizes(self):
        """Тест с разными размерами изображений."""
        test_cases = [
            (224, 16, 196),   # Стандартный ViT
            (512, 16, 1024),  # nanoVLM default
            (384, 16, 576),   # ViT-B/16 с 384
        ]

        for img_size, patch_size, expected_patches in test_cases:
            cfg = VLMConfig(
                vit_img_size=img_size,
                vit_patch_size=patch_size,
                vit_hidden_dim=768,
                vit_cls_flag=False
            )
            module = ViTPatchEmbeddings(cfg)

            x = torch.randn(1, 3, img_size, img_size)
            out = module(x)

            assert out.shape == (1, expected_patches, 768), \
                f"Для img_size={img_size}, patch_size={patch_size}: " \
                f"ожидается {expected_patches} патчей, получено {out.shape[1]}"

    def test_positional_embeddings_added(self, cfg):
        """Тест, что позиционные эмбеддинги действительно добавляются."""
        module = ViTPatchEmbeddings(cfg)

        # Создаем фейковый вход
        x = torch.randn(1, 3, 224, 224)

        # Сохраняем оригинальные позиционные эмбеддинги
        original_pos_emb = module.position_embedding.clone()

        # Forward pass
        out = module(x)

        # Позиционные эмбеддинги должны оставаться неизменными (не обучаемые в forward)
        assert torch.equal(module.position_embedding, original_pos_emb), \
            "position_embedding не должен изменяться в forward pass"

        # Проверяем, что выход не равен нулям (значит что-то было добавлено)
        assert not torch.all(out == 0), \
            "Выход равен нулю! Возможно, позиционные эмбеддинги не добавлены"


# Вспомогательная функция для быстрого запуска
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
