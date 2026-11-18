"""FOV 预处理：检测眼底 bbox → 取中心的 max(w,h) 正方形 → 越界补黑（不裁有效内容）。"""

from __future__ import annotations

import numpy as np
from PIL import Image

THR = 5
MIN_RATIO = 0.01


def detect_box(
    gray: np.ndarray, thr: int = THR, min_ratio: float = MIN_RATIO
) -> tuple[int, int, int, int] | None:
    mask = gray > thr
    row_ok = mask.mean(axis=1) >= min_ratio
    col_ok = mask.mean(axis=0) >= min_ratio
    if not row_ok.any() or not col_ok.any():
        return None
    rows = np.flatnonzero(row_ok)
    cols = np.flatnonzero(col_ok)
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def square_box(w: int, h: int, box: tuple[int, int, int, int] | None
               ) -> tuple[int, int, int]:
    """返回正方形 crop 的 (x0, y0, side)；box=None 时退回整图。"""
    if box is None:
        box = (0, 0, w, h)
    x0, y0, x1, y1 = box
    side = int(max(x1 - x0, y1 - y0))
    if side < 8:
        box = (0, 0, w, h)
        x0, y0, x1, y1 = box
        side = int(max(x1 - x0, y1 - y0))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side


def crop_to_square(img: Image.Image, thr: int = THR, min_ratio: float = MIN_RATIO
                   ) -> Image.Image:
    """返回正方形 FIG，越界区域为黑色 padding；不改变像素比例。"""
    w, h = img.size
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    box = detect_box(gray, thr, min_ratio)
    sx0, sy0, side = square_box(w, h, box)

    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    ix0, iy0 = max(0, sx0), max(0, sy0)
    ix1, iy1 = min(w, sx0 + side), min(h, sy0 + side)
    if ix1 > ix0 and iy1 > iy0:
        canvas.paste(img.crop((ix0, iy0, ix1, iy1)), (ix0 - sx0, iy0 - sy0))
    return canvas
