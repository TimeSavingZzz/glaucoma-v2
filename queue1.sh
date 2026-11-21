#!/bin/bash
# 队列 1：等 EMLONG 结束 -> 评 TTA -> 依次跑 EM3LONG、R50LONG；任一步失败即停。
# 由 nohup 常驻，日志 /mnt/glaucoma/runs/queue1.log
set -u
R=/mnt/glaucoma/runs
PY=/opt/miniconda3/envs/shadocformer/bin/python
LOG=$R/queue1.log
exec >>"$LOG" 2>&1

wait_exp() {
    while pgrep -f "train_ddp.py --exp-id $1" >/dev/null; do sleep 60; done
}

run_dir_of() {
    ls -d "$R/$1"_* 2>/dev/null | sort | tail -1
}

eval_ckpt() {  # $1=exp_id_prefix $2=npz_tag
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

run_one() {  # $1=exp $2=arch 其余为 train_ddp 参数
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

echo "=== queue1 start $(date) ==="
wait_exp V2-EMLONG
echo "[$(date)] EMLONG finished"
eval_ckpt V2-EMLONG emlong || echo "[$(date)] continue without emlong npz"

run_one V2-EM3LONG tf_efficientnetv2_m.in21k_ft_in1k \
    --size 512 --epochs 16 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02 \
    --drop-path 0.1 --aug geo2 --patience 6 --seed 2024 --swad || exit 1
eval_ckpt V2-EM3LONG em3long || true

run_one V2-R50LONG resnet50 \
    --size 512 --epochs 16 --batch-size 4 --accum 2 --lr 1e-4 --weight-decay 0.02 \
    --drop-path 0.1 --aug strong --rank-weight 0.5 --pos-weight 4 --patience 6 \
    --seed 909 --swad || exit 1
eval_ckpt V2-R50LONG r50long || true

echo "=== queue1 end $(date) ==="
