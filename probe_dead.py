"""诊断 checkpoint 输出恒定的原因。"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GlaucomaDataset
from models import build_model

CKPT = "/mnt/glaucoma/runs/V2-EM_20261118_154058/best_ap.pt"
DATA = "/mnt/glaucoma/data/glaucoma_first"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print("meta", {k: ckpt.get(k) for k in ("arch", "mean", "std", "image_size", "epoch", "ema")})

model = build_model(ckpt["arch"], num_classes=2, pretrained=False, image_size=ckpt["image_size"])
missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
ds = GlaucomaDataset(f"{DATA}/valid.csv", f"{DATA}/valid", image_size=512, train=False,
                     mean=tuple(ckpt["mean"]), std=tuple(ckpt["std"]), limit=8)
loader = DataLoader(ds, batch_size=8)
x, y, _ = next(iter(loader))
print("x stats per image mean/std:",
      [(round(float(x[i].mean()), 3), round(float(x[i].std()), 3)) for i in range(4)])
print("labels", y.tolist())

model.eval()
with torch.no_grad():
    logits = model(x)
    feats = model.forward_head(model.forward_features(x), pre_logits=True)
print("logits", logits.numpy().round(3))
print("feat std across batch", feats.std(0).mean().item(), "feat abs mean", feats.abs().mean().item())

bn_vars = [m.running_var for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
print("BN count", len(bn_vars), "var min/max",
      round(float(min(v.min() for v in bn_vars)), 6),
      round(float(max(v.max() for v in bn_vars)), 6))
w = model.get_classifier().weight
print("classifier w norm", float(w.norm()), "b", model.get_classifier().bias.detach().numpy().round(4))
