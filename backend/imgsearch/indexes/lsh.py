"""隨機超平面 LSH (Charikar, 2002)：餘弦相似度的近似最近鄰搜尋。

雜湊函數：h_r(x) = sign(r · x)，r ~ N(0, I)
關鍵性質：P[h_r(x) = h_r(y)] = 1 - θ(x, y) / π
    → 越相似的向量，越可能落在同一個桶。

把 K 個雜湊串接成一個 K-bit 的 key (AND：降低誤撞機率)，
再建 L 張獨立的表 (OR：提高找到的機率)。
兩個夾角為 θ 的向量，至少在一張表撞在一起的機率：
    P = 1 - (1 - (1 - θ/π)^K)^L

多探測 (multi-probe, Lv et al. 2007)：除了本身的桶，也查「翻轉 1 個 bit」的鄰近桶，
用較少的表達到相近的召回率。

查詢流程：取出所有候選 → 用精確的餘弦相似度重新排序 (rerank)。
實作細節：雜湊前先減去資料平均值 (centering)，見 build()。
"""
from collections import defaultdict

import numpy as np


class RandomHyperplaneLSH:
    name = "lsh"
    exact = False

    def __init__(self, n_bits=12, n_tables=8, multiprobe=1, seed=0):
        self.K, self.L, self.probe = n_bits, n_tables, multiprobe
        self.seed = seed

    def _keys(self, X):
        """(N, d) → (L, N) 的整數 key。"""
        proj = (X - self.mu) @ self.planes.reshape(self.L * self.K, -1).T            # (N, L·K)，用 BLAS 加速
        bits = (proj.reshape(len(X), self.L, self.K) > 0).astype(np.int64)
        return (bits * self.pow2).sum(-1).T                               # (L, N)

    def build(self, X):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        # 先把資料「置中」：CNN / TF-IDF 特徵全部 ≥ 0，都擠在同一個象限，
        # 通過原點的隨機超平面幾乎切不開 → 所有點落在少數幾個桶。減去平均值可解決。
        self.mu = self.X.mean(0)
        self.planes = rng.standard_normal((self.L, self.K, X.shape[1])).astype(np.float32)
        self.pow2 = (1 << np.arange(self.K)).astype(np.int64)
        keys = self._keys(self.X)
        self.tables = []
        for t in range(self.L):
            table = defaultdict(list)
            for i, key in enumerate(keys[t]):
                table[int(key)].append(i)
            self.tables.append({k: np.array(v, np.int32) for k, v in table.items()})
        return self

    def candidates(self, q):
        qk = self._keys(q[None, :].astype(np.float32))[:, 0]
        cand = []
        for t in range(self.L):
            key = int(qk[t])
            probes = [key]
            if self.probe >= 1:
                probes += [key ^ (1 << b) for b in range(self.K)]
            for p in probes:
                arr = self.tables[t].get(p)
                if arr is not None:
                    cand.append(arr)
        if not cand:
            return np.zeros(0, np.int32)
        return np.unique(np.concatenate(cand))

    def search(self, q, k=10):
        c = self.candidates(q)
        if len(c) == 0:
            return [], {"dist_computations": 0, "candidates": 0}
        sims = self.X[c] @ q.astype(np.float32)
        kk = min(k, len(c))
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[np.argsort(-sims[top], kind="stable")]
        return ([(int(c[i]), float(1 - sims[i])) for i in top],
                {"dist_computations": len(c), "candidates": len(c)})

    @staticmethod
    def collision_prob(cos_sim, K, L):
        theta = np.arccos(np.clip(cos_sim, -1, 1))
        p = 1 - theta / np.pi
        return 1 - (1 - p ** K) ** L
