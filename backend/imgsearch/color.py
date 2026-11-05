"""全域手工特徵：HSV 顏色直方圖。

8 (H) × 4 (S) × 4 (V) = 128 維。
直方圖原本適合用卡方距離，這裡取平方根 (Hellinger 轉換) 後再做 L2 正規化，
這樣「餘弦相似度 / 歐氏距離」就等價於 Hellinger 距離，可以直接套用 KD-tree、LSH。
"""
import numpy as np
from PIL import Image

BINS = (8, 4, 4)


def color_hist(img: Image.Image) -> np.ndarray:
    hsv = np.asarray(img.convert("RGB").resize((128, 128)).convert("HSV"), dtype=np.int32)
    h = hsv[..., 0] * BINS[0] // 256
    s = hsv[..., 1] * BINS[1] // 256
    v = hsv[..., 2] * BINS[2] // 256
    idx = (h * BINS[1] + s) * BINS[2] + v
    hist = np.bincount(idx.ravel(), minlength=int(np.prod(BINS))).astype(np.float32)
    hist /= hist.sum() + 1e-8
    hist = np.sqrt(hist)                       # Hellinger
    return hist / (np.linalg.norm(hist) + 1e-8)
