"""检查预训练权重与归一化下的初始 logits/CE。"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import build_model

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

arch = "tf_efficientnetv2_m.in21k_ft_in1k"
model = build_model(arch, num_classes=2, pretrained=True, image_size=512, drop_path=0.0)
print("pretrained_cfg mean/std:", model.pretrained_cfg.get("mean"), model.pretrained_cfg.get("std"))
print("num params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

p = "/mnt/glaucoma/data/glaucoma_first/valid/valid_00000.JPG"
im = Image.open(p).convert("RGB").resize((512, 512), Image.BILINEAR)
x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255).unsqueeze(0)

for name, m in (("imagenet", (x - MEAN) / STD), ("half", (x - 0.5) / 0.5)):
    with torch.no_grad():
        model.eval()
        for mode in ("eval", "train"):
            model.train(mode == "train")
            logits = model(m)
            ce0 = F.cross_entropy(logits, torch.tensor([0])).item()
            ce1 = F.cross_entropy(logits, torch.tensor([1])).item()
            print(f"{name:9s} {mode:5s} logits={logits.numpy().round(3)} ce0={ce0:.3f} ce1={ce1:.3f}")
