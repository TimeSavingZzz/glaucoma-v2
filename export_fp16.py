"""把训练 checkpoint 转成 fp16 推理权重（体积减半，≤500MB 限制）。

用法：
  python export_fp16.py --ckpt runs/V2-EM3/best_ap.pt --out model/em3_fp16.pt
只保留推理需要的字段（model/arch/image_size/mean/std/fov_crop/input_mode）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    inf = ck.get("inference_config", ck)
    sd = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in ck["model"].items()}
    out = {
        "model": sd,
        "arch": inf["arch"],
        "image_size": int(inf.get("image_size", 512)),
        "fov_crop": bool(inf.get("fov_crop", False)),
        "input_mode": str(inf.get("input_mode", "rgb")),
        "mean": list(inf.get("mean", (0.485, 0.456, 0.406))),
        "std": list(inf.get("std", (0.229, 0.224, 0.225))),
        "epoch": ck.get("epoch"),
        "metrics": ck.get("metrics"),
        "source": str(args.ckpt),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"saved {path} {path.stat().st_size / 2**20:.1f}MB arch={out['arch']} size={out['image_size']}")


if __name__ == "__main__":
    main()
