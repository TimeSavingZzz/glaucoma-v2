"""模型工厂：ResNet-50 / ConvNeXt-Tiny / EfficientNetV2-S / Swin-T / ViT-B/16（ImageNet-1K 预训练，二分类头）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    efficientnet_v2_s,
    resnet50,
    swin_t,
    vit_b_16,
)

ARCHS = ("resnet50", "convnext_tiny", "efficientnet_v2_s", "swin_t", "vit_b_16")


def _resize_vit_pos_embed(model: nn.Module, image_size: int) -> None:
    """torchvision 的 ViT 权重固定 224（14×14 patch + cls token）；这里双三次插值到目标分辨率。

    512 输入时 patch 网格是 32×32=1024，序列长度 1025，必须重算位置编码并同步改 image_size，
    否则 `_process_input` 的形状断言会直接报错。
    """
    if image_size == model.image_size:
        return
    pe = model.encoder.pos_embedding  # (1, 1+14*14, 768)
    cls, grid = pe[:, :1], pe[:, 1:]
    g = int(round(grid.shape[1] ** 0.5))
    grid = grid.reshape(1, g, g, -1).permute(0, 3, 1, 2)
    new_g = image_size // model.patch_size
    grid = F.interpolate(grid, size=(new_g, new_g), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_g * new_g, -1)
    model.encoder.pos_embedding = nn.Parameter(torch.cat([cls, grid], dim=1))
    model.image_size = image_size


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    arch: str = "resnet50",
    image_size: int = 224,
) -> nn.Module:
    if arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if arch == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if arch == "swin_t":
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if arch == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        if pretrained:
            _resize_vit_pos_embed(model, image_size)
        else:
            model.image_size = image_size
            if image_size != 224:
                g = image_size // model.patch_size
                model.encoder.pos_embedding = nn.Parameter(
                    torch.empty(1, g * g + 1, model.encoder.pos_embedding.shape[-1])
                )
                nn.init.trunc_normal_(model.encoder.pos_embedding, std=0.02)
        return model

    raise ValueError(f"未知 arch: {arch}（可选: {ARCHS}）")
