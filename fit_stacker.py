"""在 train 概率缓存上拟合 logistic 融合器，valid 只做一次性评估。

动机：valid 只有 191 个正类，直接在其上网格搜索 9+ 个权重容易过拟合噪声；
train 有 1657 个正类，在 train 上学融合权重、valid 上验证，是更稳的
learning-to-ensemble。融合器输出 = sigmoid(标准化成员 logit 的线性组合)。

用法：
  python fit_stacker.py --train_dir /mnt/glaucoma/runs/probs_train \
      --valid_dir /mnt/glaucoma/runs/probs_valid \
      --out /mnt/glaucoma/runs/stacker.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler


def zlogit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bootstrap_ap(y: np.ndarray, p: np.ndarray, n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aps = [average_precision_score(y[idx], p[idx]) for idx in
           (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            for _ in range(n))]
    return np.percentile(aps, [2.5, 97.5]).round(4).tolist()


def load_split(directory: Path, members: list[str] | None):
    files = sorted(directory.glob("*.npz"))
    if members:
        want = set(members)
        files = [f for f in files if f.stem in want]
    out, y, names = {}, None, None
    for f in files:
        d = np.load(f, allow_pickle=False)
        yy, nn = d["labels"], d["names"]
        if y is None:
            y, names = yy, nn
        elif not (np.array_equal(yy, y) and np.array_equal(nn, names)):
            raise SystemExit(f"{f} 的标签/顺序与首个成员不一致")
        out[f.stem] = d["probs"].astype(np.float64)
    if not out:
        raise SystemExit(f"{directory} 没有匹配的 npz")
    return out, y, names


def select_c(X: np.ndarray, y: np.ndarray, grid: list[float], folds: int, seed: int):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    best_c, best_ap = grid[0], -1.0
    for c in grid:
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=c, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        ap = average_precision_score(y, oof)
        print(f"  train-CV C={c:<6g} AP={ap:.5f}", flush=True)
        if ap > best_ap:
            best_c, best_ap = c, ap
    return best_c, best_ap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--valid_dir", required=True)
    ap.add_argument("--out", default="/mnt/glaucoma/runs/stacker.json")
    ap.add_argument("--members", default="", help="逗号分隔成员名（文件 stem），默认全体")
    ap.add_argument("--c_grid", default="0.003,0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    members = [m.strip() for m in args.members.split(",") if m.strip()] or None
    tr_probs, y_tr, _ = load_split(Path(args.train_dir), members)
    va_probs, y_va, names_va = load_split(Path(args.valid_dir), members)
    common = sorted(set(tr_probs) & set(va_probs))
    if not common:
        raise SystemExit("train/valid 没有共同成员")
    print(f"members={common}\ntrain n={len(y_tr)} pos={int(y_tr.sum())} | "
          f"valid n={len(y_va)} pos={int(y_va.sum())}", flush=True)

    X_tr = np.stack([zlogit(tr_probs[m]) for m in common], axis=1)
    X_va = np.stack([zlogit(va_probs[m]) for m in common], axis=1)

    print("成员 valid AP：")
    for j, m in enumerate(common):
        print(f"  {m:28s} AP={average_precision_score(y_va, X_va[:, j]):.5f}")

    grid = [float(x) for x in args.c_grid.split(",")]
    best_c, cv_ap = select_c(X_tr, y_tr, grid, args.folds, args.seed)
    print(f"选中 C={best_c:g}（train CV AP={cv_ap:.5f}）", flush=True)

    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=best_c, max_iter=5000).fit(sc.transform(X_tr), y_tr)
    p_va = clf.predict_proba(sc.transform(X_va))[:, 1]

    # 基线：等权 logit 平均
    p_eq = 1.0 / (1.0 + np.exp(-X_va.mean(axis=1)))
    ap_stack = average_precision_score(y_va, p_va)
    ap_eq = average_precision_score(y_va, p_eq)
    print(f"\nvalid stack AP={ap_stack:.6f} AUROC={roc_auc_score(y_va, p_va):.6f} "
          f"CI={bootstrap_ap(y_va, p_va)}")
    print(f"valid 等权 AP={ap_eq:.6f} AUROC={roc_auc_score(y_va, p_eq):.6f}")
    print("系数：")
    for m, w in zip(common, clf.coef_[0]):
        print(f"  {m:28s} {w:+.4f}")

    out = {
        "members": common,
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "C": best_c,
        "train_cv_ap": cv_ap,
        "valid_ap": float(ap_stack),
        "valid_auroc": float(roc_auc_score(y_va, p_va)),
        "valid_ap_ci": bootstrap_ap(y_va, p_va),
        "valid_ap_equal_weight": float(ap_eq),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez(Path(args.out).with_suffix(".npz"), probs=p_va, labels=y_va, names=names_va)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
