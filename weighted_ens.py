#!/usr/bin/env python3
"""加权 logit 融合搜索：在若干 npz 概率缓存上枚举小权重网格。"""
import itertools
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    files = sys.argv[1:]
    items, y = [], None
    for f in files:
        d = np.load(f)
        if y is None:
            y = d["labels"]
        assert np.array_equal(d["labels"], y)
        name = f.split("/")[-1].replace(".npz", "")
        items.append((name, logit(d["probs"])))
        print(f"{name:28s} AP={average_precision_score(y, d['probs']):.4f} "
              f"AUROC={roc_auc_score(y, d['probs']):.4f}")

    grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = []
    for ws in itertools.product(grid, repeat=len(items)):
        if sum(ws) == 0:
            continue
        z = sum(w * z_ for w, (_, z_) in zip(ws, items)) / sum(ws)
        p = 1 / (1 + np.exp(-z))
        res.append((average_precision_score(y, p), roc_auc_score(y, p), ws))
    res.sort(key=lambda x: -x[0])
    print("\nTOP weighted (AP):")
    for apv, aucv, ws in res[:12]:
        label = " ".join(f"{w:g}x{n}" for w, (n, _) in zip(ws, items) if w > 0)
        print(f"AP={apv:.4f} AUROC={aucv:.4f}  {label}")
    top_au = sorted(res, key=lambda x: -x[1])[:6]
    print("\nTOP weighted (AUROC):")
    for apv, aucv, ws in top_au:
        label = " ".join(f"{w:g}x{n}" for w, (n, _) in zip(ws, items) if w > 0)
        print(f"AP={apv:.4f} AUROC={aucv:.4f}  {label}")


if __name__ == "__main__":
    main()
