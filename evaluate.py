"""独立评估：加载 timm checkpoint，在 valid 上算 AP/AUROC/logloss，支持 TTA 与 bootstrap。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import GlaucomaDataset
from models import build_model

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _zoom_out(x: torch.Tensor, f: float = 0.12) -> torch.Tensor:
    """黑边回填后缩回原尺寸 = 缩小视野内容（模拟 test 图内容偏大/偏小）"""
    s = x.shape[-1]
    p = max(1, int(round(s * f)))
    b, c, h, w = x.shape
    val = torch.tensor([-m / st for m, st in zip(MEAN, STD)],
                       device=x.device, dtype=x.dtype).view(1, c, 1, 1)
    out = val.expand(b, c, h + 2 * p, w + 2 * p).clone()
    out[:, :, p:p + h, p:p + w] = x
    return F.interpolate(out, size=(s, s), mode="bilinear", align_corners=False)


def _zoom_in(x: torch.Tensor, f: float = 0.84) -> torch.Tensor:
    """中心裁剪后放大回原尺寸（模拟内容偏小需要放大）"""
    s = x.shape[-1]
    c = int(round(s * f))
    o = (s - c) // 2
    return F.interpolate(x[:, :, o:o + c, o:o + c], size=(s, s),
                         mode="bilinear", align_corners=False)


def tta_views(x: torch.Tensor, strategy: str) -> list[torch.Tensor]:
    """strategy 用 + 连接 token：h=水平翻转, v=垂直翻转, zo=zoom-out, zi=zoom-in。
    兼容旧写法 none/hflip/hv/vh。"""
    toks = set(strategy.split("+"))
    views = [x]
    h = strategy in ("hflip", "h", "hv", "vh") or "h" in toks
    v = strategy in ("hv", "vh") or "v" in toks
    if h:
        views.append(torch.flip(x, dims=[3]))
    if v:
        views.append(torch.flip(x, dims=[2]))
    if h and v:
        views.append(torch.flip(x, dims=[2, 3]))
    if "zo" in toks:
        views.append(_zoom_out(x))
    if "zi" in toks:
        views.append(_zoom_in(x))
    return views


@torch.inference_mode()
def predict(model, loader, device, strategy="none") -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    probs, labels, names = [], [], []
    for x, y, name in loader:
        x = x.to(device, non_blocking=True)
        views = tta_views(x, strategy)
        p = None
        for v in views:
            pv = torch.softmax(model(v).float(), dim=1)[:, 1]
            p = pv if p is None else p + pv
        p = p / len(views)  # 与 predict_and_cost 的视图平均口径一致
        probs.append(p.cpu())
        labels.append(y)
        names.extend(name)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), names


def bootstrap_ci(y_true, y_prob, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    aps, aucs = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yt, yp = y_true[idx], y_prob[idx]
        if yt.min() == yt.max():
            continue
        aps.append(average_precision_score(yt, yp))
        aucs.append(roc_auc_score(yt, yp))
    return {
        "ap_ci": [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
        "auroc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--valid-csv", default="")
    ap.add_argument("--valid-dir", default="")
    ap.add_argument("--strategies", default="none,hflip,hv")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--save-probs", default="")
    args = ap.parse_args()

    from config import VALID_CSV, VALID_DIR
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    inf = ckpt.get("inference_config", ckpt)
    arch = inf["arch"]
    size = args.size or int(inf.get("image_size", 512))

    mean = tuple(inf.get("mean", MEAN))
    std = tuple(inf.get("std", STD))
    ds = GlaucomaDataset(Path(args.valid_csv) if args.valid_csv else VALID_CSV,
                         Path(args.valid_dir) if args.valid_dir else VALID_DIR,
                         image_size=size, train=False,
                         fov_crop=bool(inf.get("fov_crop", False)),
                         input_mode=str(inf.get("input_mode", "rgb")),
                         mean=mean, std=std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda")
    if "." in arch:
        model = build_model(arch, num_classes=2, pretrained=False, image_size=size).to(device)
    else:  # torchvision 老权重（resnet50 / efficientnet_v2_s / ...）
        import torchvision_models
        model = torchvision_models.build_model(num_classes=2, pretrained=False,
                                               arch=arch, image_size=size).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    out = {"ckpt": args.ckpt, "arch": arch, "epoch": ckpt.get("epoch"), "size": size}
    for strategy in args.strategies.split(","):
        y_prob, y_true, names = predict(model, loader, device, strategy)
        res = {
            "ap": float(average_precision_score(y_true, y_prob)),
            "auroc": float(roc_auc_score(y_true, y_prob)),
            "logloss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        }
        res.update(bootstrap_ci(y_true, y_prob))
        out[strategy] = res
        print(strategy, json.dumps(res))
        if args.save_probs:
            stem = args.save_probs[:-4] if args.save_probs.endswith(".npz") else args.save_probs
            np.savez(f"{stem}_{strategy}.npz",
                     names=np.array(names), probs=y_prob, labels=y_true)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
