#!/bin/bash
# 队列 4：等 queue3 -> EXP-46 zoomrobust 三连（公榜尺度鲁棒性优先）-> ConvNeXt-S -> EM3LONG 第三种子。
# 日志 /mnt/glaucoma/runs/queue4.log
set -u
R=/mnt/glaucoma/runs
PY=/opt/miniconda3/envs/shadocformer/bin/python
LOG=$R/queue4.log
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
    $PY /mnt/glaucoma/proj/v2/evaluate.py --ckpt "$ckpt" \
        --strategies none,hflip,hv --batch-size 16 --workers 8 \
        --save-probs "$R/$2" || { echo "[$(date)] eval $2 FAILED"; return 1; }
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

echo "=== queue4 start $(date) ==="
while pgrep -f "queue3.sh" >/dev/null; do sleep 60; done
while pgrep -f "train_ddp.py" >/dev/null; do sleep 60; done
echo "[$(date)] queue3 finished, GPUs free"

# EXP-46 复刻：zoomrobust + ce_auc(≈CE+rank0.5) + pos_per_batch2(≈pos_weight4)
run_one V2-ZR-R50 resnet50 \
    --size 512 --epochs 8 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02 \
    --drop-path 0.1 --aug zoomrobust --rank-weight 0.5 --pos-weight 4 --patience 4 \
    --seed 42 --swad || exit 1
eval_ckpt V2-ZR-R50 zr_r50 || true

run_one V2-ZR-R50L resnet50 \
    --size 512 --epochs 16 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02 \
    --drop-path 0.1 --aug zoomrobust --rank-weight 0.5 --pos-weight 4 --patience 6 \
    --seed 43 --swad || exit 1
eval_ckpt V2-ZR-R50L zr_r50l || true

run_one V2-ZR-EM tf_efficientnetv2_m.in21k_ft_in1k \
    --size 512 --epochs 16 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02 \
    --drop-path 0.1 --aug zoomrobust --rank-weight 0.5 --pos-weight 4 --patience 6 \
    --seed 44 --swad || exit 1
eval_ckpt V2-ZR-EM zr_em || true

run_one V2-CNVS-LONG convnext_small.fb_in22k_ft_in1k \
    --size 512 --epochs 16 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02 \
    --drop-path 0.1 --aug geo2 --rank-weight 0.5 --patience 6 --seed 303 --swad || exit 1
eval_ckpt V2-CNVS-LONG cnvslong || true

run_one V2-EM3LONG3 tf_efficientnetv2_m.in21k_ft_in1k \
    --size 512 --epochs 16 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02 \
    --drop-path 0.1 --aug geo2 --patience 6 --seed 2027 --swad || exit 1
eval_ckpt V2-EM3LONG3 em3long3 || true

echo "=== queue4 end $(date) ==="
