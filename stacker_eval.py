"""在 valid 概率缓存上做 stacking，并用 5 折交叉验证估计泛化 AP/AUROC。

用法：
  python stacker_eval.py a.npz b.npz ... [--c 1.0] [--age-csv valid.csv]

成员 logit 作为特征训练 logistic 回归；用分层 5 折 CV 得到 OOF 预测再算指标，
与固定权重 logit 平均对比，避免"在 valid 上拟合再在同一条 valid 上评估"的乐观偏差。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def zlogit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bootstrap(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aps = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aps.append(average_precision_score(y[idx], p[idx]))
    return np.percentile(aps, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--age-csv", default="", help="可选：valid.csv，用 Image_name/Age 加年龄特征")
    args = ap.parse_args()

    probs, name_lists, y = [], [], None
    for f in args.files:
        d = np.load(f)
        if y is None:
            y, names_ref = d["labels"], d["names"]
        elif not np.array_equal(d["labels"], y):
            raise SystemExit(f"{f} 标签顺序不一致")
        probs.append(d["probs"])
        name_lists.append(d["names"])
        print(f"{Path(f).stem:28s} AP={average_precision_score(y, d['probs']):.4f}")

    X = np.stack([zlogit(p) for p in probs], axis=1)
    extra = []
    if args.age_csv:
        meta = pd.read_csv(args.age_csv, sep="\t", encoding="utf-8-sig", dtype=str)
        if len(meta.columns) == 1:
            meta = pd.read_csv(args.age_csv, encoding="utf-8-sig", dtype=str)
        meta.columns = [c.strip() for c in meta.columns]
        age_map = {str(n).strip().lower(): float(a) for n, a in
                   zip(meta["Image_name"], meta["Age"])}
        ages = np.array([age_map[str(n).strip().lower()] for n in names_ref])
        extra.append(np.nan_to_num(ages, nan=np.nanmedian(ages))[:, None])
        print(f"age feature: mean={np.nanmean(ages):.2f}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    feature_sets = {"logits": X}
    if extra:
        feature_sets["logits+age"] = np.concatenate([X, *extra], axis=1)
    for tag, F in feature_sets.items():
        oof = np.zeros(len(y))
        for tr, te in skf.split(F, y):
            sc = StandardScaler().fit(F[tr])
            clf = LogisticRegression(C=args.c, max_iter=5000).fit(sc.transform(F[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(F[te]))[:, 1]
        # 逐折指标反映稳定性
        fold_aps = [average_precision_score(y[te], oof[te]) for _, te in skf.split(F, y)]
        ci = bootstrap(y, oof)
        print(f"stack[{tag}] CV AP={average_precision_score(y, oof):.6f} "
              f"AUROC={roc_auc_score(y, oof):.6f} folds={np.round(fold_aps, 3).tolist()} "
              f"CI={ci.round(4).tolist()}")


if __name__ == "__main__":
    main()
