#!/bin/bash
# 队列 5：GPU 空档 -> zoomrobust 微调（从强模型低 lr 出发，兼顾 valid 与尺度鲁棒）-> 第三个 zoomrobust 种子。
# 日志 /mnt/glaucoma/runs/queue5.log
set -u
R=/mnt/glaucoma/runs
PY=/opt/miniconda3/envs/shadocformer/bin/python
LOG=$R/queue5.log
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

echo "=== queue5 start $(date) ==="
while pgrep -f "train_ddp.py" >/dev/null; do sleep 60; done
echo "[$(date)] GPUs free"

# 两阶段：先跑出强 valid 模型，再用 zoomrobust 低 lr 微调获得尺度鲁棒性
run_one V2-ZRFT-EM tf_efficientnetv2_m.in21k_ft_in1k \
    --size 512 --epochs 6 --batch-size 8 --accum 1 --lr 3e-5 --weight-decay 0.02 \
    --drop-path 0.1 --aug zoomrobust --rank-weight 0.5 --pos-weight 4 --patience 3 \
    --seed 45 --swad --init-ckpt /mnt/glaucoma/runs/V2-EM3LONG_20261121_181840/best_ap.pt || exit 1
eval_ckpt V2-ZRFT-EM zrft_em || true

run_one V2-ZRFT-R50 resnet50 \
    --size 512 --epochs 6 --batch-size 8 --accum 1 --lr 3e-5 --weight-decay 0.02 \
    --drop-path 0.1 --aug zoomrobust --rank-weight 0.5 --pos-weight 4 --patience 3 \
    --seed 46 --swad --init-ckpt /mnt/glaucoma/runs/V2-R50LONG_20261121_204041/best_ap.pt || exit 1
eval_ckpt V2-ZRFT-R50 zrft_r50 || true

# zoomrobust 从零训练的第三个种子（第一波 zr_em 单模 0.8765，看种子方差）
run_one V2-ZR-EM2 tf_efficientnetv2_m.in21k_ft_in1k \
    --size 512 --epochs 16 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02 \
    --drop-path 0.1 --aug zoomrobust --rank-weight 0.5 --pos-weight 4 --patience 6 \
    --seed 47 --swad || exit 1
eval_ckpt V2-ZR-EM2 zr_em2 || true

echo "=== queue5 end $(date) ==="
