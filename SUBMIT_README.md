# 青光眼眼底二分类 决赛提交包（v3）

## 指标（官方 valid，5,849 张，正类 191）

- **AP = 0.893305**（旧提交 0.886276，+0.0070）
- **AUROC = 0.992960**（旧提交 0.992683，+0.0003）
- 指标口径：逐模型多视图 softmax 平均 → logit → 跨模型按权平均 → sigmoid；FP16 推理，由本包 `predict_and_cost.py` 在同一 valid 上复算得到。

## 组成（4 个模型 / 7 个视图 / 约 294 MB）

| 权重文件 | 骨干 | 预训练 | 训练配置 | TTA | 权重 |
|---|---|---|---|---|---|
| `CNVS_fp16.pt` | ConvNeXt-Small | ImageNet-22k → ImageNet-1k（timm `convnext_small.fb_in22k_ft_in1k`，Apache-2.0） | 512，10 epoch，bs4×accum2×4 卡，lr 1e-4，wd 0.05，geo2 增强，EMA，按 valid AP 选模（EMA e6） | none | 1.0 |
| `EM4b_fp16.pt` | EfficientNetV2-M | ImageNet-21k → ImageNet-1k（timm `tf_efficientnetv2_m.in21k_ft_in1k`，Apache-2.0） | 512，10 epoch，bs4×accum2×4 卡，lr 1e-4，wd 0.05，geo2 增强，CE + 0.5·pairwise logistic ranking，EMA，按 valid 的 max(raw AP, EMA AP) 选模（e7，AP 0.8628） | none | 3.0 |
| `EXP09_fp16.pt` | ResNet-50 | torchvision ImageNet-1k（BSD-3） | 512，8 epoch，geo 增强，CE + 0.5·pairwise logistic | hv（4 视图） | 4.0 |
| `EXP12_fp16.pt` | EfficientNetV2-S | torchvision ImageNet-1k（BSD-3） | 512，8 epoch，geo 增强，CE，按 valid AUROC 选模 | none | 2.0 |

融合公式：每模型视图 softmax 平均 → logit；四模型 logit 按 1.0 : 3.0 : 4.0 : 2.0 加权平均；sigmoid。

权重 SHA-256：

- `CNVS_fp16.pt`：`8723432381a5778cb5abae50c67f69091e30a5c35b3af451f621acc5248c44e2`（99,019,002 B）
- `EM4b_fp16.pt`：`80af580376ec2d8cf426ef82c58a642707a4ad927205a94660da9dca7e37c439`（106,639,346 B）
- `EXP09_fp16.pt`：`a1742d84f10426cebbef270e95b3f181199c7f5891b190b6791f47d33ac98866`（47,227,006 B）
- `EXP12_fp16.pt`：`a7cfac1faf2f22e0dc5a57ae6acf36bffb4796480b46623298862aa29bbb2369`（40,909,452 B）

## 运行方式

```bash
python infer/code/predict_and_cost.py \
  --data_root <测试图片根目录> \
  --meta <meta.csv> \
  --out_dir <输出目录> \
  --num_workers 4
```

- meta 支持 Tab/逗号分隔、UTF-8 BOM、首尾空格；图片扩展名大小写不敏感。
- 输出 `<out_dir>/submission.csv`（Image_name, Probability, Pred_Label）与 `<out_dir>/cost.json`（{"avg_cost", "num_samples"}）。
- `--timing` 只做耗时测试；cost 与 predict 共用同一推理函数（本文件内）。
- 实现：同图多视图合并成 batch 前向、channels_last、FP16 权重（softmax/logit 在 FP32）、`cudnn.benchmark=False`。

## 性能（共享 3090 实测）

- 单图 7 视图（4 模型）：本机（共享 3090，含他人负载）实测 95.7 ms/张；同机旧 6 视图配方 73 ms，旧提交于空闲机器为 35.9 ms/张。
- 权重总量约 294 MB；推理显存峰值 < 3 GB。
- 官方评测卡 T4 无 TF32，因此生产路径使用 FP16；本机无 T4，最终以官方复跑为准。

## 合规说明

- 预训练权重均为**非医学** ImageNet 权重（timm Apache-2.0 / torchvision BSD-3），未使用任何医学数据集预训练、未使用外部数据。
- train/valid/test 均来自赛方提供的数据；valid 仅用于评估与选模，未参与训练；test 无标签、未参与训练与选模。
- 旧包中的 EXP-09 / EXP-12 为本团队此前训练的同赛事模型权重，按规则可继续使用；CNVS / EM4b 为本轮在赛方 train 上训练得到。
- 提交包不含任何测试集标签或预测缓存以外的数据文件。
