"""冻结骨干特征 + 逻辑回归线性探针，快速判断预训练特征质量。

用法：
  python linear_probe.py --arch vit_small_patch14_dinov2.lvd142m --size 448 \
      --limit 8000 --save-probs /mnt/glaucoma/runs/probe_dino448
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import TRAIN_CSV, TRAIN_DIR, VALID_CSV, VALID_DIR
from dataset import GlaucomaDataset
from models import data_norm


@torch.inference_mode()
def extract(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feats, labels, names = [], [], []
    for x, y, name in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=True):
            f = model(x)
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy())
        names.extend(name)
    return np.concatenate(feats), np.concatenate(labels), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="每边限制图片数（0=全量）")
    ap.add_argument("--save-probs", default="")
    args = ap.parse_args()

    mean, std = data_norm(args.arch)
    tr = GlaucomaDataset(TRAIN_CSV, TRAIN_DIR, image_size=args.size, train=False,
                         limit=args.limit, mean=mean, std=std)
    va = GlaucomaDataset(VALID_CSV, VALID_DIR, image_size=args.size, train=False,
                         mean=mean, std=std)
    device = torch.device("cuda")
    model = timm.create_model(args.arch, pretrained=True, num_classes=0,
                              img_size=args.size).to(device).eval()

    ftr, ytr, _ = extract(model, DataLoader(tr, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.workers, pin_memory=True), device)
    fva, yva, names = extract(model, DataLoader(va, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True), device)
    print(f"train {ftr.shape} pos={int(ytr.sum())}  valid {fva.shape} pos={int(yva.sum())}")

    sc = StandardScaler().fit(ftr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(ftr), ytr)
    p = clf.predict_proba(sc.transform(fva))[:, 1]
    print(json.dumps({"ap": float(average_precision_score(yva, p)),
                      "auroc": float(roc_auc_score(yva, p))}))
    if args.save_probs:
        stem = args.save_probs[:-4] if args.save_probs.endswith(".npz") else args.save_probs
        np.savez(f"{stem}.npz", names=np.array(names), probs=p, labels=yva)


if __name__ == "__main__":
    main()
