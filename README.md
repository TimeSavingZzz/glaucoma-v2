# v2：ImageNet-22k 骨干 + 4 卡 DDP

在 v1（最优 valid AP 0.886276 / AUROC 0.992683）基础上，用更强的非医学预训练骨干重训，
目标是在同一 valid 上超过原集成，并满足提交约束（≤500MB、T4 单图 ≤100ms、合规）。

## 数据

`/mnt/glaucoma/data/glaucoma_first`：

| 文件 | 样本 | 正类 |
|---|---|---|
| train.csv | 51,396 | 1,657 |
| valid.csv | 5,849 | 191 |
| test.csv | 5,855 | 无标签 |

列：`Image_name, Age, True_Label`（test 无 True_Label）；磁盘后缀 `.JPG`，CSV 为 `.jpg`。

## 环境与启动

容器 `/opt/miniconda3/envs/shadocformer`（torch 2.5.1+cu124、timm 1.0.27），4×RTX 3090。

```bash
cd /mnt/glaucoma/proj/v2
bash run_exp.sh V2-EM tf_efficientnetv2_m.in21k_ft_in1k \
  --size 512 --epochs 10 --batch-size 8 --accum 1 --lr 2e-4 \
  --weight-decay 0.05 --warmup-epochs 1 --aug geo2 --drop-path 0.1
```

`run_exp.sh` 用 torchrun 起 4 卡 DDP，日志写 `/mnt/glaucoma/runs/<EXP>_<时间戳>/`：
`train.log`、`console.log`、`metrics.csv`、`best_ap.pt`、`best_auroc.pt`、`last.pt`。

注意：不同骨干的官方归一化不同（EffNetV2 为 0.5/0.5，ConvNeXt/Swin 为 ImageNet 均值），
`models.data_norm()` 自动解析并写入 checkpoint；评估/推理按 checkpoint 记录执行。

## 实验记录（valid）

| 实验 | 骨干 | 尺寸 | 配置 | best AP | best AUROC | 备注 |
|---|---|---|---|---|---|---|
| V2-EM | tf_efficientnetv2_m.in21k_ft_in1k | 512 | bs8×4、10ep、lr2e-4、geo2、EMA | 训练中 | | 2026-11-18 15:41 启动 |

## 合规要点

- 权重全部为 ImageNet-1k/22k 预训练（timm），Apache-2.0/MIT，非医学、非 NC、非 gated。
- 训练仅用官方 train；valid 仅评估/选模；test 仅最终推理。
- 推理：BN eval、固定种子、无随机增强；cost 与 predict 共用 `predict_prob()`。
- 最终加载权重（去重）≤500MB；T4 单图平均 ≤100ms。
