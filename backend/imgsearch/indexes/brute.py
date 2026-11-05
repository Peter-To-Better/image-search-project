"""暴力線性掃描 (Linear Scan)：精確解，作為所有近似方法的 baseline。

時間：O(N·d)    空間：O(N·d)
"""
import numpy as np

from ..hashing import hamming_many


class BruteForceHamming:
    name = "brute"
    exact = True

    def build(self, hashes):
        self.arr = np.array(hashes, dtype=np.uint64)
        return self

    def search(self, q: int, k: int = 10):
        d = hamming_many(q, self.arr)
        k = min(k, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        idx = idx[np.lexsort((idx, d[idx]))]
        return [(int(i), float(d[i])) for i in idx], {"dist_computations": len(d)}


class BruteForceCosine:
    name = "brute"
    exact = True

    def build(self, X):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        return self

    def search(self, q, k: int = 10):
        sims = self.X @ q.astype(np.float32)
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx], kind="stable")]
        return [(int(i), float(1 - sims[i])) for i in idx], {"dist_computations": len(sims)}
