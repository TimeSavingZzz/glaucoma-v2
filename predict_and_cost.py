"""统一提交入口：混合 timm / torchvision 权重，多模型 + TTA，输出 submission.csv 与 cost.json。

与训练代码共享预处理（ImageNet 归一化、可选 FOV 裁剪），推理端：
- 同一模型的多 TTA 视图合并成一个 batch 前向，视图 softmax 平均；
- 跨模型对 logit 加权平均（默认等权），再 sigmoid；
- FP16 权重 + channels_last；cost.py 与 predict 使用同一推理函数（本文件 --timing 模式）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import torchvision_models
from models import build_model as build_timm_model

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_TO_TENSOR = transforms.Compose([transforms.ToTensor()])  # 归一化在 GPU 上按模型配置做

# (权重路径, TTA 策略, 融合权重) —— valid AP 0.8934 / AUROC 0.9930，7 视图最终配方
MODELS: tuple[tuple[str, str, float], ...] = (
    ("CNVS_fp16.pt", "none", 1.0),
    ("EM4b_fp16.pt", "none", 3.0),
    ("EXP09_fp16.pt", "hv", 4.0),
    ("EXP12_fp16.pt", "none", 2.0),
)
MODEL_DIR = Path(__file__).resolve().parent / "model"
USE_FP16 = True


def active_models() -> tuple[tuple[str, str, float], ...]:
    """环境变量 GLAUCOMA_MODELS="路径::tta::权重,..." 可临时覆盖。"""
    spec = os.environ.get("GLAUCOMA_MODELS", "").strip()
    if not spec:
        return MODELS
    out = []
    for item in spec.split(","):
        parts = [p.strip() for p in item.split("::")]
        name, tta = parts[0], (parts[1] if len(parts) > 1 and parts[1] else "none")
        weight = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
        out.append((name, tta, weight))
    return tuple(out)


def index_images(root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_file():
            index[p.name.lower()] = str(p.relative_to(root))
    return index


def read_meta(path: str | Path) -> pd.DataFrame:
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str)
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        if "Image_name" in df.columns:
            df["Image_name"] = df["Image_name"].astype(str).str.strip()
            return df
    raise SystemExit(f"无法解析 meta: {path}")


def load_image_tensor(path, size, fov_crop=False, input_mode="rgb") -> torch.Tensor:
    from fov import crop_to_square
    with Image.open(path) as im:
        im.draft("RGB", (1024, 1024) if fov_crop else (size, size))
        im = im.convert("RGB")
        if input_mode == "green":
            im = im.getchannel("G").convert("RGB")
        elif input_mode == "gray":
            im = im.convert("L").convert("RGB")
        if fov_crop:
            im = crop_to_square(im)
        im = im.resize((size, size), Image.Resampling.BILINEAR)
        return _TO_TENSOR(im)


class ImagePathDataset(Dataset):
    def __init__(self, paths, size, fov_crop, input_mode):
        self.paths, self.size = paths, size
        self.fov_crop, self.input_mode = fov_crop, input_mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return load_image_tensor(self.paths[idx], self.size, self.fov_crop, self.input_mode)


def _infer_config(ckpt: dict) -> dict:
    if "inference_config" in ckpt:
        inf = ckpt["inference_config"]
        return {"arch": inf["arch"], "image_size": int(inf["image_size"]),
                "fov_crop": bool(inf.get("fov_crop", False)),
                "input_mode": str(inf.get("input_mode", "rgb")),
                "mean": tuple(inf.get("mean", MEAN)), "std": tuple(inf.get("std", STD))}
    return {"arch": ckpt["arch"], "image_size": int(ckpt.get("image_size", 512)),
            "fov_crop": bool(ckpt.get("fov_crop", False)),
            "input_mode": str(ckpt.get("input_mode", "rgb")),
            "mean": tuple(ckpt.get("mean", MEAN)), "std": tuple(ckpt.get("std", STD))}


def build_any(arch: str, image_size: int) -> nn.Module:
    if "." in arch:
        return build_timm_model(arch, num_classes=2, pretrained=False, image_size=image_size)
    return torchvision_models.build_model(num_classes=2, pretrained=False,
                                          arch=arch, image_size=image_size)


def load_models(model_dir: Path = MODEL_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, strategies, weights, infos, norms = [], [], [], [], []
    for name, tta, weight in active_models():
        path = Path(name) if Path(name).is_absolute() else model_dir / name
        if not path.is_file():
            raise SystemExit(f"缺少权重: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        inf = _infer_config(ckpt)
        model = build_any(inf["arch"], inf["image_size"]).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model = model.to(memory_format=torch.channels_last)
        if USE_FP16 and device.type == "cuda":
            model = model.half()
        model.eval()
        models.append(model)
        strategies.append(tta)
        weights.append(weight)
        infos.append(inf)
        norms.append((torch.tensor(inf["mean"], device=device).view(1, 3, 1, 1),
                      torch.tensor(inf["std"], device=device).view(1, 3, 1, 1)))
    first = infos[0]
    for inf in infos[1:]:
        for k in ("image_size", "fov_crop", "input_mode"):
            if inf[k] != first[k]:
                raise SystemExit(f"权重输入配置不一致: {k} {inf[k]} vs {first[k]}")
    return models, strategies, first, norms, weights


def views(x: torch.Tensor, tta: str) -> list[torch.Tensor]:
    if tta == "none":
        return [x]
    if tta == "hflip":
        return [x, TF.hflip(x)]
    if tta == "hv":
        hf = TF.hflip(x)
        return [x, hf, TF.vflip(x), TF.vflip(hf)]
    raise ValueError(f"未知 TTA: {tta}")


def predict_prob(models, strategies, norms, x: torch.Tensor, weights=None) -> torch.Tensor:
    """x 为未归一化的 [0,1] 图像；每个模型按自身预训练配置归一化。"""
    if weights is None:
        weights = [1.0] * len(models)
    model_logits, wsum = [], 0.0
    for model, strategy, (mean, std), w in zip(models, strategies, norms, weights):
        vs = [(v - mean) / std for v in views(x, strategy)]
        dtype = next(model.parameters()).dtype
        batch = torch.cat(vs, dim=0).to(dtype=dtype).contiguous(memory_format=torch.channels_last)
        probs = torch.softmax(model(batch).float(), dim=1)[:, 1].view(len(vs), x.size(0))
        model_logits.append(w * torch.logit(probs.mean(dim=0).clamp(1e-6, 1 - 1e-6)))
        wsum += w
    return torch.sigmoid(sum(model_logits) / wsum)


def resolve_paths(data_root: Path, names: list[str]) -> list[Path]:
    index, paths, missing = None, [], []
    for n in names:
        direct = data_root / n
        if direct.is_file():
            paths.append(direct)
            continue
        if index is None:
            index = index_images(data_root)
        key = Path(n).name.lower()
        if key in index:
            paths.append(data_root / index[key])
        else:
            missing.append(n)
    if missing:
        raise SystemExit(f"缺图 {len(missing)} 张，示例 {missing[:3]}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./test")
    ap.add_argument("--meta", default="./test.csv")
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", "--workers", dest="workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n_timed", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timing", action="store_true", help="只跑耗时测试")
    args = ap.parse_args()

    data_root, out_dir = Path(args.data_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = read_meta(args.meta)["Image_name"].astype(str).tolist()
    paths = resolve_paths(data_root, names)

    models, strategies, inf, norms, weights = load_models()
    size = inf["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    with torch.inference_mode():
        for p in paths[: args.warmup]:
            x = load_image_tensor(p, size, inf["fov_crop"], inf["input_mode"]).unsqueeze(0).to(device)
            predict_prob(models, strategies, norms, x, weights)
    torch.cuda.synchronize()

    rest = list(range(len(paths)))[args.warmup:]
    rng = random.Random(args.seed)
    timed_idx = rng.sample(rest, min(args.n_timed, len(rest)))
    costs = []
    with torch.inference_mode():
        for i in timed_idx:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = load_image_tensor(paths[i], size, inf["fov_crop"], inf["input_mode"]).unsqueeze(0).to(device)
            p = predict_prob(models, strategies, norms, x, weights)
            torch.cuda.synchronize()
            costs.append((time.perf_counter() - t0) * 1000.0)
            _ = float(p[0])
    avg_cost = float(np.mean(costs))
    print(f"avg_cost {avg_cost:.3f} ms  p50={np.percentile(costs, 50):.1f} "
          f"p95={np.percentile(costs, 95):.1f} max={max(costs):.1f}")
    with open(out_dir / "cost.json", "w", encoding="utf-8") as f:
        json.dump({"avg_cost": round(avg_cost, 3), "num_samples": len(costs)}, f)

    if args.timing:
        return

    loader = DataLoader(ImagePathDataset(paths, size, inf["fov_crop"], inf["input_mode"]),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    probs = []
    with torch.inference_mode():
        for x in loader:
            probs.append(predict_prob(models, strategies, norms, x.to(device, non_blocking=True), weights).cpu().numpy())
    prob = np.concatenate(probs)
    frame = pd.DataFrame({"Image_name": names,
                          "Probability": np.round(prob, 9),
                          "Pred_Label": (prob >= args.threshold).astype(np.int64)})
    frame.to_csv(out_dir / "submission.csv", index=False, encoding="utf-8",
                 float_format="%.9f", lineterminator="\n")
    print(f"images {len(names)}  positive {int(frame['Pred_Label'].sum())} "
          f"({frame['Pred_Label'].mean():.4%})")
    print(f"output {out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
