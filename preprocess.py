"""纯 numpy 的 CLAHE（限制对比度自适应直方图均衡），训练与推理共用、逐位可复现。

对眼底图取绿通道后做 CLAHE 再复制成三通道：绿通道对渗出/血管/杯盘边界对比度最高，
CLAHE 补偿不同相机的曝光差异。实现为分块直方图 + 双线性插值，避免 cv2 依赖。
"""

from __future__ import annotations

import numpy as np

N_TILES = (8, 8)
CLIP_LIMIT = 2.0


def _clip_hist(hist: np.ndarray, clip: float) -> np.ndarray:
    """按 CLAHE 规则裁剪直方图，溢出部分均匀回填。"""
    excess = int(np.maximum(hist - clip, 0).sum())
    hist = np.minimum(hist, clip)
    if excess:
        hist += excess // hist.size
        hist[: excess % hist.size] += 1
    return hist


def _build_luts(gray: np.ndarray, n_tiles: tuple[int, int], clip_limit: float) -> np.ndarray:
    ny, nx = n_tiles
    H, W = gray.shape
    th, tw = H // ny, W // nx
    luts = np.zeros((ny, nx, 256), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            block = gray[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            hist = np.bincount(block.ravel(), minlength=256).astype(np.float32)
            clip = max(1.0, clip_limit * block.size / 256.0)
            cdf = np.cumsum(_clip_hist(hist, clip))
            luts[i, j] = cdf / max(cdf[-1], 1.0) * 255.0
    return luts


def clahe_gray(gray: np.ndarray, n_tiles: tuple[int, int] = N_TILES,
               clip_limit: float = CLIP_LIMIT) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    h, w = gray.shape
    ny, nx = n_tiles
    ph, pw = (-h) % ny, (-w) % nx
    g = np.pad(gray, ((0, ph), (0, pw)), mode="reflect")
    H, W = g.shape
    th, tw = H // ny, W // nx
    luts = _build_luts(g, n_tiles, clip_limit)

    ys = np.arange(H) / th - 0.5
    xs = np.arange(W) / tw - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, ny - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    wy = np.clip(ys - y0, 0.0, 1.0)[:, None]
    wx = np.clip(xs - x0, 0.0, 1.0)[None, :]

    idx = g.astype(np.int64)
    v00 = luts[y0[:, None], x0[None, :], idx]
    v01 = luts[y0[:, None], x1[None, :], idx]
    v10 = luts[y1[:, None], x0[None, :], idx]
    v11 = luts[y1[:, None], x1[None, :], idx]
    out = (v00 * (1 - wx) + v01 * wx) * (1 - wy) + (v10 * (1 - wx) + v11 * wx) * wy
    return np.clip(out, 0, 255).astype(np.uint8)[:h, :w]


def clahe_rgb(im):
    """PIL RGB 图 → 绿通道 CLAHE → 复制三通道的 PIL 图。"""
    from PIL import Image

    green = np.asarray(im.getchannel("G"))
    out = clahe_gray(green)
    return Image.fromarray(out, mode="L").convert("RGB")
