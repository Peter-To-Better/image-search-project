"""感知雜湊 (Perceptual Hashing) —— 全部自己實作，不依賴 imagehash 套件。

每個雜湊都是 64 bit，存成 Python int，兩張圖的距離 = Hamming 距離
(兩個整數 XOR 之後 1 的個數)。

  aHash : 縮成 8x8 灰階，像素 > 平均值 記 1
  dHash : 縮成 9x8 灰階，比較左右相鄰像素 (水平梯度) 的大小
  pHash : 縮成 32x32 灰階 → 2D DCT → 取左上角 8x8 低頻係數 → 與中位數比較
"""
import numpy as np
from PIL import Image


def _gray(img: Image.Image, w: int, h: int) -> np.ndarray:
    return np.asarray(img.convert("L").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float64)


def _bits_to_int(bits: np.ndarray) -> int:
    v = 0
    for b in bits.flatten():
        v = (v << 1) | int(b)
    return v


def ahash(img: Image.Image) -> int:
    p = _gray(img, 8, 8)
    return _bits_to_int(p > p.mean())


def dhash(img: Image.Image) -> int:
    p = _gray(img, 9, 8)
    return _bits_to_int(p[:, 1:] > p[:, :-1])


# ---- DCT：自己用矩陣乘法實作 (DCT-II, orthonormal) ----
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


_D32 = _dct_matrix(32)


def dct2(x: np.ndarray) -> np.ndarray:
    """2D DCT = D · X · Dᵀ，時間複雜度 O(n³)（n=32 時可忽略）。"""
    return _D32 @ x @ _D32.T


def phash(img: Image.Image) -> int:
    p = _gray(img, 32, 32)
    low = dct2(p)[:8, :8].flatten()
    med = np.median(low[1:])          # 排除 DC 係數 (整體亮度)，對亮度變化較穩健
    return _bits_to_int(low > med)


if hasattr(int, "bit_count"):          # Python 3.10+
    def hamming(a: int, b: int) -> int:
        return (a ^ b).bit_count()
else:                                   # Python 3.9 相容
    def hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")


HASHERS = {"ahash": ahash, "dhash": dhash, "phash": phash}


def hashes_to_array(hs) -> np.ndarray:
    """list[int] → (N,) uint64，給向量化的暴力搜尋用。"""
    return np.array(hs, dtype=np.uint64)


def hamming_many(q: int, arr: np.ndarray) -> np.ndarray:
    """一次計算 q 與 N 個雜湊的 Hamming 距離 (向量化)。"""
    x = np.bitwise_xor(arr, np.uint64(q))
    if hasattr(np, "bitwise_count"):          # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int32)
    b = x.view(np.uint8).reshape(-1, 8)
    return np.unpackbits(b, axis=1).sum(axis=1).astype(np.int32)
