#!/bin/bash
# 队列 6：queue5 结束后接续。目标：增加可部署（低时延）强成员与尺度轴。
# 日志 /mnt/glaucoma/runs/queue6.log
set -u
R=/mnt/glaucoma/runs
PY=/opt/miniconda3/envs/shadocformer/bin/python
LOG=$R/queue6.log
exec >>"$LOG" 2>&1

wait_exp() {
    while pgrep -f "train_ddp.py --exp-id $1" >/dev/null; do sleep 60; done
}

run_dir_of() {
    ls -d "$R/$1"_* 2>/dev/null | sort | tail -1
}

eval_ckpt() {
    local rd ckpt
    rd=$(run_dir_of "$1")
    ckpt="$rd/best_ap.pt"
    if [ ! -f "$ckpt" ]; then
        echo "[$(date)] MISSING $ckpt"
        return 1
    fi
    $PY /mnt/glaucoma/proj/v2/evaluate.py --ckpt "$ckpt"         --strategies none,hflip,hv --batch-size 16 --workers 8         --save-probs "$R/$2" || { echo "[$(date)] eval $2 FAILED"; return 1; }
    return 0
}

run_one() {
    local exp=$1 arch=$2 out rd
    shift 2
    out=$(bash /mnt/glaucoma/proj/v2/run_exp.sh "$exp" "$arch" "$@") || {
        echo "[$(date)] launch $exp FAILED"; return 1; }
    echo "[$(date)] $out"
    rd=$(echo "$out" | sed -n 's/.*run=\([^ ]*\).*/\1/p')
    sleep 300
    if ! pgrep -f "train_ddp.py --exp-id $exp" >/dev/null; then
        echo "[$(date)] $exp died within 5 min"
        return 1
    fi
    wait_exp "$exp"
    if grep -q "DONE" "$rd/train.log" 2>/dev/null; then
        echo "[$(date)] $exp DONE ($rd)"
        return 0
    fi
    echo "[$(date)] $exp ended WITHOUT DONE ($rd)"
    return 1
}

echo "=== queue6 start $(date) ==="
while pgrep -f "queue5.sh" >/dev/null; do sleep 60; done
while pgrep -f "train_ddp.py" >/dev/null; do sleep 60; done
echo "[$(date)] GPUs free"

# 1) r50long 第三个种子（512，strong+posw4，复刻 0.88238 配方）
run_one V2-R50LONG3 resnet50     --size 512 --epochs 16 --batch-size 4 --accum 2 --lr 1e-4 --weight-decay 0.02     --drop-path 0.1 --aug strong --rank-weight 0.5 --pos-weight 4 --patience 6     --seed 48 --swad || exit 1
eval_ckpt V2-R50LONG3 r50long3 || true

# 2) r50@640（新尺度轴，视图成本约 1.5x，仍可部署）
run_one V2-R50-640 resnet50     --size 640 --epochs 12 --batch-size 4 --accum 2 --lr 1e-4 --weight-decay 0.02     --drop-path 0.1 --aug strong --rank-weight 0.5 --pos-weight 4 --patience 5     --seed 50 --swad || exit 1
eval_ckpt V2-R50-640 r50_640 || true

# 3) em3long 第四个种子（geo2，最强单模 0.8864 的配方）
run_one V2-EM3LONG4 tf_efficientnetv2_m.in21k_ft_in1k     --size 512 --epochs 16 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02     --drop-path 0.1 --aug geo2 --rank-weight 0.5 --pos-weight 1 --patience 6     --seed 53 --swad || exit 1
eval_ckpt V2-EM3LONG4 em3long4 || true

echo "=== queue6 end $(date) ==="
