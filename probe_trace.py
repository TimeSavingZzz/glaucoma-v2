"""逐层检查激活是否塌缩、权重是否含 NaN。"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GlaucomaDataset
from models import build_model

CKPT = "/mnt/glaucoma/runs/V2-EM_20261118_154058/last.pt"
DATA = "/mnt/glaucoma/data/glaucoma_first"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model"]
bad = [(k, float(torch.isnan(v).float().mean()), float(torch.isinf(v).float().mean()))
       for k, v in sd.items() if v.dtype.is_floating_point]
bad = [b for b in bad if b[1] > 0 or b[2] > 0]
print("nan/inf tensors:", bad[:10], "count", len(bad))
zero = [(k, float((v == 0).float().mean())) for k, v in sd.items() if v.ndim == 4]
print("conv zero-ratio worst:", sorted(zero, key=lambda x: -x[1])[:5])

m = build_model(ck["arch"], num_classes=2, pretrained=False, image_size=512)
m.load_state_dict(sd)
m.eval()

ds = GlaucomaDataset(f"{DATA}/valid.csv", f"{DATA}/valid", image_size=512, train=False,
                     mean=tuple(ck["mean"]), std=tuple(ck["std"]), limit=8)
x, y, _ = next(iter(DataLoader(ds, batch_size=8)))
print("input img std (per-image then mean):", float(x.std(dim=(1, 2, 3)).mean()))

acts = []
hooks = []
for name, mod in m.named_children():
    def hook(mod, inp, out, name=name):
        if isinstance(out, torch.Tensor):
            acts.append((name, tuple(out.shape), float(out.std(dim=0).mean()),
                         float(out.abs().mean())))
    hooks.append(mod.register_forward_hook(hook))
with torch.no_grad():
    m(x)
for a in acts:
    print(a)
