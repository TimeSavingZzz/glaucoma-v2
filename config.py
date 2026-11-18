"""全局配置：路径全部可由环境变量覆盖，避免硬编码。"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("GLAUCOMA_DATA", "/mnt/glaucoma/data/glaucoma_first"))
RUNS_DIR = Path(os.environ.get("GLAUCOMA_RUNS", "/mnt/glaucoma/runs"))

TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_DIR = DATA_ROOT / "train"
VALID_CSV = DATA_ROOT / "valid.csv"
VALID_DIR = DATA_ROOT / "valid"
TEST_CSV = DATA_ROOT / "test.csv"
TEST_DIR = DATA_ROOT / "test"

IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 8
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("negative", "positive")
