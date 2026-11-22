#!/bin/bash
# 等 ZR-EM2 训练结束 -> 对 X 包成员做尺度 TTA 评测（o=zoom-out, i=zoom-in）
set -u
R=/mnt/glaucoma/runs
PY=/opt/miniconda3/envs/shadocformer/bin/python
LOG=$R/scale_eval.log
exec >>"$LOG" 2>&1

echo "=== scale_eval start $(date) ==="
while pgrep -f "train_ddp.py --exp-id V2-ZR-EM2" >/dev/null; do sleep 30; done
echo "[$(date)] ZR-EM2 finished, start evals"

$PY /mnt/glaucoma/proj/v2/evaluate.py   --ckpt $R/V2-R50LONG_20261121_204041/best_ap.pt   --strategies none,hflip,hv,o,i,io --batch-size 32 --workers 8   --save-probs $R/r50long

$PY /mnt/glaucoma/proj/v2/evaluate.py   --ckpt $R/V2-EM4b_20261118_235510/best_ap.pt   --strategies none,hflip,hv,o,i,io --batch-size 32 --workers 8   --save-probs $R/em4b

$PY /mnt/glaucoma/proj/v2/evaluate.py   --ckpt $R/V2-CNVS_20261118_203123/best_ap.pt   --strategies none,hflip,hv,o,i,io --batch-size 32 --workers 8   --save-probs $R/cnvs

echo "=== scale_eval end $(date) ==="
