#!/bin/bash
# 队列 7：温和 zoom-out 尺度增强（只缩小黑边，不做放大裁剪），验证能否在不掉 valid 的前提下拿到尺度鲁棒性。
set -u
R=/mnt/glaucoma/runs
PY=/opt/miniconda3/envs/shadocformer/bin/python
LOG=$R/queue7.log
exec >>"$LOG" 2>&1

wait_exp() { while pgrep -f "train_ddp.py --exp-id $1" >/dev/null; do sleep 60; done; }
run_dir_of() { ls -d "$R/$1"_* 2>/dev/null | sort | tail -1; }

eval_ckpt() {
    local rd ckpt
    rd=$(run_dir_of "$1")
    ckpt="$rd/best_ap.pt"
    [ -f "$ckpt" ] || { echo "[$(date)] MISSING $ckpt"; return 1; }
    $PY /mnt/glaucoma/proj/v2/evaluate.py --ckpt "$ckpt"         --strategies none,hflip,hv --batch-size 16 --workers 8         --save-probs "$R/$2" || { echo "[$(date)] eval $2 FAILED"; return 1; }
}

run_one() {
    local exp=$1 arch=$2 out rd
    shift 2
    out=$(bash /mnt/glaucoma/proj/v2/run_exp.sh "$exp" "$arch" "$@") || {
        echo "[$(date)] launch $exp FAILED"; return 1; }
    echo "[$(date)] $out"
    rd=$(echo "$out" | sed -n 's/.*run=\([^ ]*\).*/\1/p')
    sleep 300
    pgrep -f "train_ddp.py --exp-id $exp" >/dev/null || { echo "[$(date)] $exp died"; return 1; }
    wait_exp "$exp"
    grep -q "DONE" "$rd/train.log" 2>/dev/null && { echo "[$(date)] $exp DONE ($rd)"; return 0; }
    echo "[$(date)] $exp ended WITHOUT DONE ($rd)"; return 1
}

echo "=== queue7 start $(date) ==="
while pgrep -f "queue6.sh" >/dev/null; do sleep 30; done
while pgrep -f "evaluate.py" >/dev/null; do sleep 30; done
while pgrep -f "train_ddp.py" >/dev/null; do sleep 30; done
echo "[$(date)] GPUs free"

run_one V2-ZO-EM tf_efficientnetv2_m.in21k_ft_in1k     --size 512 --epochs 16 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02     --drop-path 0.1 --aug zoomout --rank-weight 0.5 --pos-weight 1 --patience 6     --seed 55 --swad || exit 1
eval_ckpt V2-ZO-EM zo_em || true

run_one V2-ZO-R50 resnet50     --size 512 --epochs 16 --batch-size 4 --accum 2 --lr 1e-4 --weight-decay 0.02     --drop-path 0.1 --aug zoomoutstrong --rank-weight 0.5 --pos-weight 4 --patience 6     --seed 56 --swad || exit 1
eval_ckpt V2-ZO-R50 zo_r50 || true

run_one V2-ZO-CNVS convnext_small.fb_in22k_ft_in1k     --size 512 --epochs 16 --batch-size 8 --accum 1 --lr 1e-4 --weight-decay 0.02     --drop-path 0.1 --aug zoomout --rank-weight 0.5 --pos-weight 1 --patience 6     --seed 58 --swad || exit 1
eval_ckpt V2-ZO-CNVS zo_cnvs || true

echo "=== queue7 end $(date) ==="
