"""
Configuration classes for nanoVLM components.

Централизованная конфигурация всех гиперпараметров модели.
"""

from dataclasses import dataclass


@dataclass
class VLMConfig:
    """
    Configuration for Vision-Language Model.

    Attributes:
        Vision Transformer (ViT) parameters:
            vit_hidden_dim: Размерность эмбеддингов ViT
            vit_inter_dim: Размерность intermediate слоя в MLP
            vit_patch_size: Размер патча (например, 16x16)
            vit_img_size: Размер входного изображения
            vit_n_heads: Количество attention heads
            vit_dropout: Dropout rate
            vit_n_blocks: Количество transformer блоков
            vit_ln_eps: Epsilon для LayerNorm
            vit_cls_flag: Использовать ли CLS token

        Language Model (LM) parameters:
            lm_hidden_dim: Размерность эмбеддингов LM
            lm_inter_dim: Размерность intermediate слоя в MLP
            lm_rms_eps: Epsilon для RMSNorm
            lm_re_base: Base для Rotary Embeddings
            lm_max_position_embeddings: Максимальная длина последовательности
            lm_vocab_size: Размер словаря (base + extra tokens)
            lm_n_heads: Количество query heads
            lm_n_kv_heads: Количество key-value heads (для GQA)
            lm_dropout: Dropout rate
            lm_n_blocks: Количество decoder блоков
            lm_max_length: Максимальная длина для generation

        Modality Projector parameters:
            mp_pixel_shuffle_factor: Фактор сжатия для pixel shuffle
            mp_image_token_length: Финальная длина последовательности для изображения
    """

    # Vision Transformer
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim  # 3072
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False

    # Language Model
    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66  # Специальные токены для VLM
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5  # Grouped Query Attention
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_max_length: int = 4096

    # Modality Projector
    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64


@dataclass
class TrainConfig:
    """
    Training configuration.

    Содержит все параметры для обучения модели:
    learning rates, batch size, optimizer settings, etc.
    """

    # Learning rates (разные для разных компонентов)
    lr_mp: float = 0.00512  # Modality Projector (обучается с нуля)
    lr_vision_backbone: float = 5e-5  # ViT (fine-tuning)
    lr_language_backbone: float = 5e-5  # LM (fine-tuning)

    # Training parameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    max_training_steps: int = 40000
    max_sample_length: int = 4096

    # Evaluation
    eval_interval: int = 500
    stats_log_interval: int = 100
