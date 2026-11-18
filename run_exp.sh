#!/bin/bash
# 4 卡 DDP 训练启动器
# 用法: bash run_exp.sh <EXP-ID> <ARCH> [train_ddp.py 其余参数...]
# 例:   bash run_exp.sh V2-EM tf_efficientnetv2_m.in21k_ft_in1k --size 512 --epochs 10 \
#           --batch-size 4 --accum 2 --lr 2e-4
set -eu
EXP=$1
ARCH=$2
shift 2

PY=/opt/miniconda3/envs/shadocformer/bin
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/glaucoma/proj/v2
STAMP=$(date +%Y%m%d_%H%M%S)
RUN=/mnt/glaucoma/runs/${EXP}_${STAMP}
mkdir -p "$RUN"
PORT=$((29500 + RANDOM % 500))

nohup "$PY/torchrun" --nproc_per_node=4 --master_port="$PORT" \
    train_ddp.py --exp-id "$EXP" --arch "$ARCH" --run-dir "$RUN" "$@" \
    > "$RUN/console.log" 2>&1 &

echo "STARTED pid=$! run=$RUN port=$PORT"
