"""局部特徵 + 視覺詞袋 (Bag of Visual Words)。

流程：
 1. 每張圖用 SIFT 抽出數百個 128 維局部描述子 (RootSIFT 版本)
 2. 用 k-means 把所有描述子分成 K 群 → 每一群就是一個「視覺單字」
 3. 每張圖 = 視覺單字的出現次數直方圖 → TF-IDF 加權 → L2 正規化
 4. 搜尋時用「倒排索引」(inverted index)，跟文字搜尋引擎同樣的做法
"""
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

MAX_SIDE = 320
N_FEATURES = 250


def _to_cv(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    w, h = g.size
    s = MAX_SIDE / max(w, h)
    if s < 1:
        g = g.resize((max(1, int(w * s)), max(1, int(h * s))))
    return np.asarray(g, dtype=np.uint8)


_sift = None


def sift_descriptors(img: Image.Image) -> np.ndarray:
    """回傳 (n, 128) uint8 的原始 SIFT 描述子。"""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    _, d = _sift.detectAndCompute(_to_cv(img), None)
    if d is None:
        return np.zeros((0, 128), np.uint8)
    return np.clip(d, 0, 255).astype(np.uint8)


def root_sift(d: np.ndarray) -> np.ndarray:
    """RootSIFT：L1 正規化後開根號 (Arandjelović & Zisserman, 2012)。"""
    d = d.astype(np.float32)
    d /= d.sum(axis=1, keepdims=True) + 1e-7
    return np.sqrt(d)


class BoVW:
    def __init__(self, k: int = 1000):
        self.k = k
        self.kmeans = None
        self.idf = None

    def fit(self, desc_list, sample: int = 150_000, seed: int = 42):
        rng = np.random.default_rng(seed)
        allv = np.concatenate([d for d in desc_list if len(d)], axis=0)
        if len(allv) > sample:
            allv = allv[rng.choice(len(allv), sample, replace=False)]
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=4096, n_init=3,
                                      random_state=seed, max_iter=100)
        self.kmeans.fit(root_sift(allv))
        # 計算 IDF
        df = np.zeros(self.k, np.float64)
        for d in desc_list:
            if len(d):
                df[np.unique(self.kmeans.predict(root_sift(d)))] += 1
        self.idf = np.log((len(desc_list) + 1) / (df + 1)).astype(np.float32)
        return self

    def encode(self, d: np.ndarray) -> np.ndarray:
        v = np.zeros(self.k, np.float32)
        if len(d):
            words = self.kmeans.predict(root_sift(d))
            v = np.bincount(words, minlength=self.k).astype(np.float32)
            v /= v.sum()
            v *= self.idf
            v /= np.linalg.norm(v) + 1e-8
        return v

    def state(self):
        return {"centers": self.kmeans.cluster_centers_.astype(np.float32), "idf": self.idf}

    @classmethod
    def from_state(cls, centers, idf):
        b = cls(k=len(centers))
        b.kmeans = _NearestCenter(centers)
        b.idf = idf
        return b


class _NearestCenter:
    """載入後只需要「找最近中心」，自己實作避免依賴 sklearn 內部狀態。"""

    def __init__(self, centers):
        self.cluster_centers_ = centers.astype(np.float32)
        self._c2 = (self.cluster_centers_ ** 2).sum(1)

    def predict(self, x):
        x = x.astype(np.float32)
        d = self._c2[None, :] - 2 * x @ self.cluster_centers_.T
        return d.argmin(1)
