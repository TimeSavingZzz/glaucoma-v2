"""预训练模型在 valid 图像上的特征多样性检查。"""
import torch
from torch.utils.data import DataLoader

from dataset import GlaucomaDataset
from models import build_model, data_norm

arch = "tf_efficientnetv2_m.in21k_ft_in1k"
mean, std = data_norm(arch)
m = build_model(arch, num_classes=2, pretrained=True, image_size=512).eval()
ds = GlaucomaDataset("/mnt/glaucoma/data/glaucoma_first/valid.csv",
                     "/mnt/glaucoma/data/glaucoma_first/valid",
                     image_size=512, train=False, mean=mean, std=std, limit=8)
x, y, _ = next(iter(DataLoader(ds, batch_size=8)))
with torch.no_grad():
    f = m.forward_head(m.forward_features(x), pre_logits=True)
    lg = m(x)
print("feat std across batch", float(f.std(0).mean()), "abs", float(f.abs().mean()))
print("logit std across batch", float(lg.std(0).mean()))
print(lg[:4].numpy().round(2))
