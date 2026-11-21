"""比较几种与校准无关的融合方式，用于补充加权 logit 平均。

- logit_mean：原始 logit 等权平均（当前做法）
- z_logit_mean：每个成员先在全体样本上做 z-score 再平均（消除 logit 尺度差异）
- rank_mean：每个成员的平均秩/n 后平均（完全与单调变换无关）
AP 只看排序，这三种都可以直接在 valid 上比较。

用法：python rank_ens.py a.npz b.npz ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def zlogit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bootstrap_ap(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aps = [average_precision_score(y[i], p[i]) for i in
           (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            for _ in range(n))]
    return np.percentile(aps, [2.5, 97.5]).round(4).tolist()


def main() -> None:
    files = sys.argv[1:]
    probs, y = [], None
    for f in files:
        d = np.load(f)
        if y is None:
            y = d["labels"]
        elif not np.array_equal(d["labels"], y):
            raise SystemExit(f"{f} 标签顺序不一致")
        probs.append(d["probs"])
        print(f"{Path(f).stem:28s} AP={average_precision_score(y, d['probs']):.5f}")
    L = np.stack([zlogit(p) for p in probs], axis=1)
    R = np.stack([rankdata(L[:, j]) / len(y) for j in range(L.shape[1])], axis=1)

    cands = {
        "logit_mean": L.mean(axis=1),
        "z_logit_mean": ((L - L.mean(0)) / L.std(0)).mean(axis=1),
        "rank_mean": R.mean(axis=1),
        "logit_median": np.median(L, axis=1),
        "rank_median": np.median(R, axis=1),
    }
    for name, s in cands.items():
        print(f"{name:14s} AP={average_precision_score(y, s):.6f} "
              f"AUROC={roc_auc_score(y, s):.6f} CI={bootstrap_ap(y, s)}")


if __name__ == "__main__":
    main()
