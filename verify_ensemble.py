"""用 predict_and_cost 的真实推理管线在 valid 上复算 AP/AUROC。

用法：
  GLAUCOMA_MODELS="a.pt::hv,b.pt::hflip" python verify_ensemble.py --limit 0
确认提交管线的数值与 ensemble_search 的 npz 融合一致。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

import predict_and_cost as pc
from config import VALID_CSV, VALID_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(VALID_DIR))
    ap.add_argument("--meta", default=str(VALID_CSV))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = pc.read_meta(args.meta)
    names = meta["Image_name"].astype(str).tolist()
    labels = meta["True_Label"].astype(int).to_numpy()
    if args.limit:
        names, labels = names[: args.limit], labels[: args.limit]
    paths = pc.resolve_paths(Path(args.data_root), names)

    models, strategies, infos, norms, weights = pc.load_models()
    sizes = sorted({inf["image_size"] for inf in infos})
    ds = pc.MultiSizeDataset(paths, sizes, infos[0]["fov_crop"], infos[0]["input_mode"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    dev = next(models[0].parameters()).device
    probs = []
    with torch.inference_mode():
        for batch in loader:
            xs = {s: t.to(dev, non_blocking=True) for s, t in batch.items()}
            probs.append(pc.predict_prob(models, strategies, infos, norms, xs,
                                         weights).cpu().numpy())
    p = np.concatenate(probs)
    print(f"models={len(models)} strategies={strategies} n={len(p)}")
    print(f"AP={average_precision_score(labels, p):.6f} "
          f"AUROC={roc_auc_score(labels, p):.6f}")
    out = Path("output")
    out.mkdir(exist_ok=True)
    np.savez(out / "valid_probs.npz", names=np.array(names), probs=p, labels=labels)


if __name__ == "__main__":
    main()
