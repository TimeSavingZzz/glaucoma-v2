"""模型 soup：同架构多个 checkpoint 权重平均后在 valid 上评估。

用法：
  python soup_eval.py --ckpts runs/A/best_ap.pt runs/B/best_ap.pt --strategies none,hv \
      --save-probs /mnt/glaucoma/runs/soup_ab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader

from dataset import GlaucomaDataset
from evaluate import predict
from models import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--strategies", default="none,hv")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-probs", default="")
    args = ap.parse_args()

    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in args.ckpts]
    infos = [c.get("inference_config", c) for c in ckpts]
    inf = infos[0]
    for other in infos[1:]:
        for k in ("arch", "image_size", "fov_crop", "input_mode"):
            if other.get(k, inf.get(k)) != inf.get(k):
                raise SystemExit(f"soup 配置不一致: {k}")
    arch = inf["arch"]
    size = int(inf.get("image_size", 512))

    soups = {}
    keys = ckpts[0]["model"].keys()
    for k in keys:
        vals = [c["model"][k] for c in ckpts]
        if vals[0].dtype.is_floating_point:
            soups[k] = torch.stack([v.float() for v in vals]).mean(0).to(vals[0].dtype)
        else:
            soups[k] = vals[0]

    from config import VALID_CSV, VALID_DIR
    ds = GlaucomaDataset(VALID_CSV, VALID_DIR, image_size=size, train=False,
                         fov_crop=bool(inf.get("fov_crop", False)),
                         input_mode=str(inf.get("input_mode", "rgb")),
                         mean=tuple(inf.get("mean", (0.485, 0.456, 0.406))),
                         std=tuple(inf.get("std", (0.229, 0.224, 0.225))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda")
    if "." in arch:
        model = build_model(arch, num_classes=2, pretrained=False, image_size=size).to(device)
    else:
        import torchvision_models
        model = torchvision_models.build_model(num_classes=2, pretrained=False,
                                               arch=arch, image_size=size).to(device)
    model.load_state_dict(soups, strict=True)

    for strategy in args.strategies.split(","):
        y_prob, y_true, names = predict(model, loader, device, strategy)
        res = {
            "ap": float(average_precision_score(y_true, y_prob)),
            "auroc": float(roc_auc_score(y_true, y_prob)),
            "logloss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        }
        print(strategy, json.dumps(res))
        if args.save_probs:
            stem = args.save_probs[:-4] if args.save_probs.endswith(".npz") else args.save_probs
            np.savez(f"{stem}_{strategy}.npz", names=np.array(names), probs=y_prob, labels=y_true)


if __name__ == "__main__":
    main()
