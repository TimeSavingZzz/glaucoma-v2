"""数据管线：大小写不敏感索引 + JPEG draft 解码 + 几何/颜色增强。"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD
from fov import crop_to_square

AUG_MODES = ("basic", "geo", "geo2", "zoom")


def _index_images(image_dir: Path) -> dict[str, str]:
    """磁盘文件名可能大写 .JPG，CSV 是小写 .jpg；递归索引以兼容按子目录存放。"""
    index: dict[str, str] = {}
    for path in image_dir.rglob("*"):
        if path.is_file():
            index[path.name.lower()] = str(path.relative_to(image_dir))
    return index


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class GlaucomaDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        image_size: int = 512,
        train: bool = False,
        check_files: bool = True,
        limit: int = 0,
        aug: str = "geo",
        fov_crop: bool = False,
        input_mode: str = "rgb",
        name_col: str = "Image_name",
        label_col: str = "True_Label",
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.fov_crop = bool(fov_crop)
        if aug not in AUG_MODES:
            raise ValueError(f"未知 aug 模式: {aug}（可选: {AUG_MODES}）")
        self.aug = aug
        if input_mode not in ("rgb", "green", "gray"):
            raise ValueError(f"未知 input_mode: {input_mode}")
        self.input_mode = input_mode

        df = pd.read_csv(self.csv_path)
        if name_col not in df.columns:
            raise ValueError(f"{self.csv_path} 缺少 {name_col} 列")
        self.names = df[name_col].astype(str).tolist()
        self.has_label = label_col in df.columns
        self.labels = df[label_col].astype(int).tolist() if self.has_label else [-1] * len(self.names)

        if limit and limit > 0:
            self.names = self.names[:limit]
            self.labels = self.labels[:limit]

        index = _index_images(self.image_dir)
        missing = [name for name in self.names if name.lower() not in index]
        if check_files and missing:
            raise FileNotFoundError(
                f"{self.image_dir} 缺少 {len(missing)} 个文件，示例: {missing[:3]}"
            )
        self.paths = [str(self.image_dir / index[name.lower()]) for name in self.names]

        if self.train:
            if self.aug == "basic":
                ops = [transforms.RandomHorizontalFlip(p=0.5)]
            elif self.aug == "geo":
                ops = [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=10, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=0
                    ),
                ]
            elif self.aug == "geo2":
                ops = [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=12, translate=(0.03, 0.03), scale=(0.93, 1.07),
                        shear=3, fill=0,
                    ),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                ]
            else:  # zoom
                ops = [
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.6, 1.0), ratio=(0.9, 1.1),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=10, translate=(0.02, 0.02), scale=(0.98, 1.02), fill=0
                    ),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                ]
            self.transform = transforms.Compose(
                ops + [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )

    def __len__(self) -> int:
        return len(self.names)

    def _load_image(self, path: str) -> Image.Image:
        zoom = self.train and self.aug == "zoom"
        target = 1024 if (self.fov_crop or zoom) else self.image_size
        with Image.open(path) as im:
            im.draft("RGB", (target, target))
            im = im.convert("RGB")
            if self.input_mode == "green":
                im = im.getchannel("G").convert("RGB")
            elif self.input_mode == "gray":
                im = im.convert("L").convert("RGB")
            if self.fov_crop:
                im = crop_to_square(im)
            side = 1024 if zoom else self.image_size
            im = im.resize((side, side), Image.Resampling.BILINEAR)
        return im

    def __getitem__(self, idx: int):
        image = self.transform(self._load_image(self.paths[idx]))
        return image, int(self.labels[idx]), self.names[idx]

    def label_counts(self) -> dict[int, int]:
        if not self.has_label:
            return {}
        values, counts = np.unique(np.asarray(self.labels), return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}


class BoundedPosBatchSampler:
    """每个 batch 固定 pos_per_batch 个正类（仅单机用，DDP 下未启用）。"""

    def __init__(self, labels, batch_size, pos_per_batch, seed=42):
        self.pos = [i for i, y in enumerate(labels) if int(y) == 1]
        self.neg = [i for i, y in enumerate(labels) if int(y) == 0]
        if not self.pos or not self.neg:
            raise ValueError("需要同时存在正负样本")
        if pos_per_batch >= batch_size:
            raise ValueError("pos_per_batch 必须小于 batch_size")
        self.batch_size = int(batch_size)
        self.pos_per_batch = int(pos_per_batch)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        n_neg = self.batch_size - self.pos_per_batch
        return max(math.ceil(len(self.pos) / self.pos_per_batch),
                   math.ceil(len(self.neg) / n_neg))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        n_neg = self.batch_size - self.pos_per_batch
        pos_pool, neg_pool = [], []
        neg_order = rng.permutation(len(self.neg))
        pos_order = rng.permutation(len(self.pos))

        def take(pool, ids, k, order):
            while len(pool) < k:
                pool.extend(ids[i] for i in order)
            out = pool[:k]
            del pool[:k]
            return out

        for _ in range(len(self)):
            batch = take(pos_pool, self.pos, self.pos_per_batch, pos_order)
            batch += take(neg_pool, self.neg, n_neg, neg_order)
            yield [batch[i] for i in rng.permutation(len(batch))]


def build_loader(csv_path, image_dir, image_size, batch_size, train, num_workers=8,
                 seed=42, limit=0, aug="geo", fov_crop=False, input_mode="rgb",
                 pos_per_batch=0, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    dataset = GlaucomaDataset(csv_path, image_dir, image_size=image_size, train=train,
                              limit=limit, aug=aug, fov_crop=fov_crop, input_mode=input_mode,
                              mean=mean, std=std)
    if train and pos_per_batch > 0:
        sampler = BoundedPosBatchSampler(dataset.labels, batch_size, pos_per_batch, seed)
        return dataset, DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                                   pin_memory=True, persistent_workers=num_workers > 0,
                                   worker_init_fn=_worker_init_fn)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return dataset, DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        pin_memory=True, drop_last=train, persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn, generator=generator if train else None,
    )
