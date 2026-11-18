"""预下载候选骨干的 ImageNet-22k 权重到 HF 缓存（CPU 即可，不占显存）。"""
import timm

ARCHS = [
    "convnext_base.fb_in22k_ft_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnext_small.fb_in22k_ft_in1k",
    "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
]
for arch in ARCHS:
    try:
        try:
            timm.create_model(arch, pretrained=True, num_classes=2, img_size=512)
        except TypeError:
            timm.create_model(arch, pretrained=True, num_classes=2)
        print("OK", arch, flush=True)
    except Exception as e:  # noqa: BLE001
        print("FAIL", arch, repr(e), flush=True)
