"""比较 logit / 概率 / 秩 / z-score 四种加权融合方式。

用法：
  python fusion_lab.py a.npz b.npz ... --weights 1 3 4 2
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def zlogit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def report(name, y, p):
    print(f"{name:10s} AP={average_precision_score(y, p):.6f} "
          f"AUROC={roc_auc_score(y, p):.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    args = ap.parse_args()

    probs, y = [], None
    for f in args.files:
        d = np.load(f)
        if y is None:
            y = d["labels"]
        p = d["probs"]
        probs.append(p)
        print(f"{f.split('/')[-1]:40s} AP={average_precision_score(y, p):.4f}")
    w = np.asarray(args.weights if args.weights else [1.0] * len(probs), dtype=float)
    w = w / w.sum()

    report("logit", y, 1 / (1 + np.exp(-sum(wi * zlogit(p) for wi, p in zip(w, probs)))))
    report("prob", y, sum(wi * p for wi, p in zip(w, probs)))
    report("rank", y, sum(wi * rankdata(p) / len(p) for wi, p in zip(w, probs)))
    z = np.stack([(zlogit(p) - zlogit(p).mean()) / zlogit(p).std() for p in probs])
    report("zscore", y, 1 / (1 + np.exp(-(w[:, None] * z).sum(0))))


if __name__ == "__main__":
    main()
