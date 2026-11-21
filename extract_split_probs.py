"""对 train/valid 全量跑各成员模型（含 TTA）概率，供 train 上拟合融合器。

与 predict_and_cost.py 共用图像解码/预处理/TTA 逻辑，保证推理口径一致。
每个成员存一个 npz：probs / labels / names / member（路径::tta）。

用法：
  python extract_split_probs.py --split valid \
      --members "pkg_v3/model/CNVS_fp16.pt::none,/mnt/glaucoma/runs/V2-EM4b_.../best_ap.pt::hv" \
      --out_dir /mnt/glaucoma/runs/probs_valid --batch_size 32 --workers 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TRAIN_CSV, TRAIN_DIR, VALID_CSV, VALID_DIR
from predict_and_cost import (MultiSizeDataset, _infer_config, build_any, load_image_tensor,
                              read_meta, resolve_paths, views)


def member_id(path: str, tta: str) -> str:
    return f"{Path(path).stem}_{tta}".replace(".", "_")


@torch.inference_mode()
def extract(model_path: str, tta: str, paths, batch_size: int, workers: int, device):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    inf = _infer_config(ckpt)
    size = int(inf["image_size"])
    model = build_any(inf["arch"], size).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    if device.type == "cuda":
        model = model.half()
    model = model.to(memory_format=torch.channels_last).eval()

    ds = MultiSizeDataset(paths, [size], inf["fov_crop"], inf["input_mode"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    mean = torch.tensor(inf["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(inf["std"], device=device).view(1, 3, 1, 1)
    dtype = next(model.parameters()).dtype
    out = []
    for batch in loader:
        xb = batch[size].to(device, non_blocking=True)
        vs = [(v - mean) / std for v in views(xb, tta)]
        b = torch.cat(vs, dim=0).to(dtype=dtype).contiguous(memory_format=torch.channels_last)
        p = torch.softmax(model(b).float(), dim=1)[:, 1].view(len(vs), xb.size(0))
        out.append(p.mean(dim=0).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "valid"], required=True)
    ap.add_argument("--members", required=True,
                    help="逗号分隔 path::tta，例如 a.pt::none,b.pt::hv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    if args.split == "train":
        csv_path, data_root = TRAIN_CSV, TRAIN_DIR
    else:
        csv_path, data_root = VALID_CSV, VALID_DIR
    meta = read_meta(csv_path)
    if args.limit:
        meta = meta.head(args.limit)
    names = meta["Image_name"].astype(str).tolist()
    labels = meta["True_Label"].astype(int).to_numpy()
    paths = resolve_paths(Path(data_root), names)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"split={args.split} n={len(names)} pos={int(labels.sum())} device={device}",
          flush=True)

    for spec in [s.strip() for s in args.members.split(",") if s.strip()]:
        if "::" in spec:
            model_path, tta = spec.split("::", 1)
        else:
            model_path, tta = spec, "none"
        mid = member_id(model_path, tta)
        dst = out_dir / f"{mid}.npz"
        if args.skip_existing and dst.is_file():
            print(f"skip {mid}", flush=True)
            continue
        probs = extract(model_path, tta, paths, args.batch_size, args.workers, device)
        np.savez(dst, probs=probs, labels=labels, names=np.array(names),
                 member=spec)
        print(f"{mid:28s} AP={average_precision(labels, probs):.4f} "
              f"mean={probs.mean():.4f} -> {dst}", flush=True)


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


if __name__ == "__main__":
    main()
