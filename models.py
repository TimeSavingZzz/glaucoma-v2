"""timm 模型工厂（ImageNet / ImageNet-22k 非医学预训练权重）。"""

from __future__ import annotations

import timm
from timm.models import get_pretrained_cfg

FALLBACK_MEAN = (0.485, 0.456, 0.406)
FALLBACK_STD = (0.229, 0.224, 0.225)

# 统一候选：全部为 ImageNet-1k 或 ImageNet-22k 预训练，符合赛题第 1 条
ARCHS = (
    "convnext_base.fb_in22k_ft_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k",
    "tf_efficientnetv2_m.in21k_ft_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_small.fb_in22k_ft_in1k",
    "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
)


def build_model(arch: str, num_classes: int = 2, pretrained: bool = True,
                image_size: int = 512, drop_path: float = 0.1, drop_rate: float = 0.0):
    kwargs = dict(pretrained=pretrained, num_classes=num_classes,
                  drop_path_rate=drop_path, drop_rate=drop_rate)
    try:
        model = timm.create_model(arch, img_size=image_size, **kwargs)
    except TypeError:
        model = timm.create_model(arch, **kwargs)
    return model


def data_norm(arch: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """返回该骨干官方预训练配置的 mean/std（不同家族不一致，必须匹配）。"""
    try:
        cfg = get_pretrained_cfg(arch)
        mean = tuple(cfg.mean) if cfg.mean else FALLBACK_MEAN
        std = tuple(cfg.std) if cfg.std else FALLBACK_STD
        return mean, std
    except Exception:  # noqa: BLE001
        return FALLBACK_MEAN, FALLBACK_STD
