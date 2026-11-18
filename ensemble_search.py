"""在 valid 概率缓存上搜索等权 logit 融合组合。

用法：
  python ensemble_search.py probs/*.npz --top 20
每个 npz 需包含 probs / labels / names（evaluate.py --save-probs 产物）。
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bootstrap(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aps, aucs = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aps.append(average_precision_score(y[idx], p[idx]))
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aps, [2.5, 97.5]), np.percentile(aucs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--max-models", type=int, default=4)
    args = ap.parse_args()

    items = []
    y_ref, names_ref = None, None
    for f in args.files:
        d = np.load(f)
        p, y, names = d["probs"], d["labels"], d["names"]
        if y_ref is None:
            y_ref, names_ref = y, names
        elif not (np.array_equal(y, y_ref) and np.array_equal(names, names_ref)):
            raise SystemExit(f"{f} 的样本顺序与其他文件不一致")
        items.append((Path(f).stem, p))
        print(f"{Path(f).stem:48s} AP={average_precision_score(y, p):.4f} "
              f"AUROC={roc_auc_score(y, p):.4f}")

    results = []
    for k in range(1, min(args.max_models, len(items)) + 1):
        for combo in itertools.combinations(range(len(items)), k):
            z = np.mean([logit(items[i][1]) for i in combo], axis=0)
            p = 1 / (1 + np.exp(-z))
            results.append((average_precision_score(y_ref, p), roc_auc_score(y_ref, p),
                            combo, p))
    results.sort(key=lambda x: -x[0])
    print("\nTOP ensembles (equal-weight logit):")
    for apv, aucv, combo, p in results[: args.top]:
        label = " + ".join(items[i][0] for i in combo)
        print(f"AP={apv:.4f} AUROC={aucv:.4f}  {label}")
    best = results[0][3]
    ci_ap, ci_au = bootstrap(y_ref, best)
    print(f"\nbest AP 95% CI {ci_ap.round(4).tolist()}  AUROC CI {ci_au.round(4).tolist()}")


if __name__ == "__main__":
    main()
