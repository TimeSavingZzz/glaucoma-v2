"""在部署约束（模型数 ≤5、TTA 视图 ≤8、fp16 权重 ≤500MB）下搜索最优融合。

- 每个 npz 是一个"模型+一种 TTA"的候选；同一模型的多个 TTA 变体不能同时入选；
- 枚举模型子集（去重），权重网格 {1,2,3}，先按 fit 集 AP 排序；
- 取 fit 前若干名在 eval 集上评估（fit=eval 时为乐观估计，仅作探索）。

用法：
  python deploy_ens.py --fit_dir /mnt/glaucoma/runs --eval_dir /mnt/glaucoma/runs
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# npz stem -> (模型标识, TTA 视图数, fp16 权重 MB)
SPEC = {
    "cnvs_none": ("cnvs", 1, 95), "cnvs_hflip": ("cnvs", 2, 95), "cnvs_hv": ("cnvs", 4, 95),
    "cnxb384_none": ("cnxb384", 1, 168), "cnxb384_hv": ("cnxb384", 4, 168),
    "em3_hflip": ("em3", 2, 102), "em3_hv": ("em3", 4, 102),
    "em4b_none": ("em4b", 1, 102), "em4b_hflip": ("em4b", 2, 102), "em4b_hv": ("em4b", 4, 102),
    "em4b640_none": ("em4b640", 1, 102), "em4b640_hv": ("em4b640", 4, 102),
    "em5ovs_none": ("em5ovs", 1, 102), "em5ovs_hv": ("em5ovs", 4, 102),
    "old09_hflip": ("old09", 2, 45), "old09_hv": ("old09", 4, 45),
    "old09_640_hv": ("old09_640", 4, 45),
    "old12_none": ("old12", 1, 42), "old12_hflip": ("old12", 2, 42), "old12_hv": ("old12", 4, 42),
    "r50ovs_none": ("r50ovs", 1, 45), "r50ovs_hv": ("r50ovs", 4, 45),
    "r50fov_none": ("r50fov", 1, 45), "r50fov_hv": ("r50fov", 4, 45),
    "r50grn_none": ("r50grn", 1, 45), "r50grn_hv": ("r50grn", 4, 45),
    "swinb384_none": ("swinb384", 1, 166), "swinb384_hv": ("swinb384", 4, 166),
    "soup_em4b_none": ("soup", 1, 102), "soup_em4b_hv": ("soup", 4, 102),
}

MAX_MODELS = 5
MAX_VIEWS = 8
MAX_MB = 500
WEIGHTS = (1, 2, 3)


def zlogit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# 弱成员/冗余成员不进入搜索（cnxb384、swin、soup、r50grn 单模明显弱）
EXCLUDE = {"cnxb384_none", "cnxb384_hv", "swinb384_none", "swinb384_hv",
           "soup_em4b_none", "soup_em4b_hv", "r50grn_none", "r50grn_hv"}


def load_dir(directory: Path):
    data = {}
    for npz in sorted(directory.glob("*.npz")):
        if npz.stem not in SPEC or npz.stem in EXCLUDE:
            continue
        d = np.load(npz)
        data[npz.stem] = (zlogit(d["probs"].astype(np.float64)), d["labels"])
    return data


def bootstrap_ap(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aps = [average_precision_score(y[i], p[i]) for i in
           (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            for _ in range(n))]
    return np.percentile(aps, [2.5, 97.5]).round(4).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_dir", default="/mnt/glaucoma/runs")
    ap.add_argument("--eval_dir", default="")
    ap.add_argument("--top_fit", type=int, default=60)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    fit = load_dir(Path(args.fit_dir))
    evald = fit if not args.eval_dir else load_dir(Path(args.eval_dir))
    stems = [s for s in fit if s in evald]
    if not stems:
        raise SystemExit("没有共同 npz")

    # 按模型去重：每个模型只保留一个 TTA 变体作为候选
    combos = []
    for r in range(1, MAX_MODELS + 1):
        for subset in itertools.combinations(stems, r):
            models = [SPEC[s][0] for s in subset]
            if len(set(models)) != len(models):
                continue
            views = sum(SPEC[s][1] for s in subset)
            mb = sum(SPEC[s][2] for s in subset)
            if views > MAX_VIEWS or mb > MAX_MB:
                continue
            combos.append(subset)
    print(f"候选组合 {len(combos)} 个（≤{MAX_MODELS} 模型, ≤{MAX_VIEWS} 视图, ≤{MAX_MB}MB）")

    fit_res = []
    for subset in combos:
        Ls = [fit[s][0] for s in subset]
        y = fit[subset[0]][1]
        best = None
        for ws in itertools.product(WEIGHTS, repeat=len(subset)):
            z = sum(w * l for w, l in zip(ws, Ls)) / sum(ws)
            a = average_precision_score(y, z)
            if best is None or a > best[0]:
                best = (a, ws)
        fit_res.append((best[0], subset, best[1]))
    fit_res.sort(key=lambda t: -t[0])
    print(f"fit 集最优 AP={fit_res[0][0]:.6f}（乐观，选型在 fit 上完成）")

    print(f"\nfit 前 {args.show} 名在 eval 集上的表现：")
    rows = []
    for a_fit, subset, ws in fit_res[: args.top_fit]:
        y = evald[subset[0]][1]
        z = sum(w * evald[s][0] for w, s in zip(ws, subset)) / sum(ws)
        rows.append((average_precision_score(y, z), roc_auc_score(y, z), a_fit, subset, ws))
    rows.sort(key=lambda t: -t[0])
    for ap_eval, auc_eval, a_fit, subset, ws in rows[: args.show]:
        views = sum(SPEC[s][1] for s in subset)
        mb = sum(SPEC[s][2] for s in subset)
        print(f"AP_eval={ap_eval:.6f} AUROC={auc_eval:.6f} AP_fit={a_fit:.6f} "
              f"views={views} MB={mb} :: "
              + " ".join(f"{w}x{s}" for w, s in zip(ws, subset)))
    best = rows[0]
    y = evald[best[3][0]][1]
    z = sum(w * evald[s][0] for w, s in zip(best[4], best[3])) / sum(best[4])
    print(f"\n最佳 eval 组合 bootstrap CI={bootstrap_ap(y, z)}")


if __name__ == "__main__":
    main()
