"""部署约束下的 beam search 融合（模型 ≤5、TTA 视图 ≤8、fp16 权重 ≤500MB）。

穷举权重网格在成员变多后组合爆炸；改用按 AP 排序的束搜索：
每层给状态追加一个成员（权重 1/2/3，同一模型的不同 TTA 变体不能同时入选），
保留 AP 最高的 K 个状态进入下一层。状态分值用 sum(w*logit)（AP 对单调变换不变）。

用法：python beam_ens.py --dir /mnt/glaucoma/runs [--beam 40]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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
    "emlong_none": ("emlong", 1, 102), "emlong_hflip": ("emlong", 2, 102),
    "emlong_hv": ("emlong", 4, 102),
    "em3long_none": ("em3long", 1, 102), "em3long_hflip": ("em3long", 2, 102),
    "em3long_hv": ("em3long", 4, 102),
    "r50long_none": ("r50long", 1, 45), "r50long_hflip": ("r50long", 2, 45),
    "r50long_hv": ("r50long", 4, 45),
    "exp09_640_none": ("exp09_640", 1, 45), "exp09_640_hflip": ("exp09_640", 2, 45),
    "exp09_640_hv": ("exp09_640", 4, 45),
    "clahe_em_none": ("clahe_em", 1, 102), "clahe_em_hflip": ("clahe_em", 2, 102),
    "clahe_em_hv": ("clahe_em", 4, 102),
}
EXCLUDE = {"cnxb384_none", "cnxb384_hv", "swinb384_none", "swinb384_hv",
           "soup_em4b_none", "soup_em4b_hv", "r50grn_none", "r50grn_hv",
           "exp09_640_none", "exp09_640_hflip", "exp09_640_hv"}

MAX_MODELS, MAX_VIEWS, MAX_MB = 5, 8, 500
WEIGHTS = (1, 2, 3)


def zlogit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load(directory: Path):
    out = {}
    for f in sorted(directory.glob("*.npz")):
        if f.stem in SPEC and f.stem not in EXCLUDE:
            d = np.load(f)
            out[f.stem] = (zlogit(d["probs"].astype(np.float64)), d["labels"])
    return out


def bootstrap_ap(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aps = [average_precision_score(y[i], p[i]) for i in
           (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            for _ in range(n))]
    return np.percentile(aps, [2.5, 97.5]).round(4).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/glaucoma/runs")
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    data = load(Path(args.dir))
    stems = list(data)
    y = data[stems[0]][1]
    print(f"候选 {len(stems)} 个，valid n={len(y)} pos={int(y.sum())}")

    # 状态: (view_count, mb, used_models frozenset, [(stem, weight)], z)
    beam = [(-1.0, 0, 0, frozenset(), (), None)]
    seen = []
    for depth in range(1, MAX_MODELS + 1):
        nxt = []
        for _, views, mb, used, comp, z in beam:
            for s in stems:
                model, v, m = SPEC[s]
                if model in used or views + v > MAX_VIEWS or mb + m > MAX_MB:
                    continue
                for w in WEIGHTS:
                    zn = w * data[s][0] if z is None else z + w * data[s][0]
                    a = average_precision_score(y, zn)
                    nxt.append((a, views + v, mb + m, used | {model}, comp + ((s, w),), zn))
        neval = len(nxt)
        # 去重：同一 (组件集合) 只留最高分
        name = {}
        for st in nxt:
            key = tuple(sorted(st[4]))
            if key not in name or st[0] > name[key][0]:
                name[key] = st
        uniq = sorted(name.values(), key=lambda t: -t[0])[: args.beam]
        seen.extend(uniq)
        print(f"depth {depth}: 评估 {neval} 状态，保留 {len(uniq)}，"
              f"本层最佳 AP={uniq[0][0]:.6f} :: "
              + " ".join(f"{w}x{s}" for s, w in uniq[0][4]), flush=True)
        beam = [(t[0], t[1], t[2], t[3], t[4], t[5]) for t in uniq]
        if not beam:
            break

    final = {}
    for st in seen:
        key = tuple(sorted(st[4]))
        if key not in final or st[0] > final[key][0]:
            final[key] = st
    rows = sorted(final.values(), key=lambda t: -t[0])[: args.show]
    print("\n最终前排名单（同一 valid，选型乐观）：")
    for a, views, mb, used, comp, z in rows:
        print(f"AP={a:.6f} AUROC={roc_auc_score(y, z):.6f} views={views} MB={mb} :: "
              + " ".join(f"{w}x{s}" for s, w in comp))
    top = rows[0]
    print(f"\n最佳 bootstrap CI={bootstrap_ap(y, top[5])}")


if __name__ == "__main__":
    main()
