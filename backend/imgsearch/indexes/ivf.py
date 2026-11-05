"""IVF (Inverted File Index)：先分群，查詢時只搜尋最近的幾群。

建立：用 k-means (自己實作的 Lloyd 演算法，含 k-means++ 初始化) 把 N 個向量分成 nlist 群。
查詢：1) 算查詢點到 nlist 個中心的距離，取最近的 nprobe 群   O(nlist·d)
      2) 只在這些群內暴力搜尋                                  O(N·nprobe/nlist·d)
nlist ≈ √N 時，總成本約 O(√N·d)。
缺點：查詢點落在群邊界時，真正的最近鄰可能在沒被探測的群裡。
"""
import numpy as np


def kmeans(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    # k-means++ 初始化
    centers = [X[rng.integers(n)]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        c = X[rng.choice(n, p=d2 / d2.sum())]
        centers.append(c)
        d2 = np.minimum(d2, ((X - c) ** 2).sum(1))
    C = np.stack(centers).astype(np.float32)
    for _ in range(iters):
        assign = (-(X @ C.T) * 2 + (C ** 2).sum(1)).argmin(1)          # E-step
        newC = np.zeros_like(C)
        np.add.at(newC, assign, X)
        cnt = np.bincount(assign, minlength=k)[:, None]
        empty = cnt[:, 0] == 0
        newC[~empty] /= cnt[~empty]
        newC[empty] = X[rng.choice(n, empty.sum())]                    # 空群重新抽樣
        if np.allclose(newC, C, atol=1e-5):
            break
        C = newC                                                       # M-step
    return C, assign


class IVF:
    name = "ivf"
    exact = False

    def __init__(self, nlist=None, nprobe=8, seed=0):
        self.nlist, self.nprobe, self.seed = nlist, nprobe, seed

    def build(self, X):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        nlist = self.nlist or max(4, int(np.sqrt(len(X))))
        self.C, assign = kmeans(self.X, nlist, seed=self.seed)
        self.lists = [np.where(assign == c)[0].astype(np.int32) for c in range(nlist)]
        return self

    def search(self, q, k=10):
        q = q.astype(np.float32)
        cs = self.C @ q
        probe = np.argsort(-cs)[: self.nprobe]
        cand = np.concatenate([self.lists[c] for c in probe])
        if len(cand) == 0:
            return [], {"dist_computations": len(self.C), "candidates": 0}
        sims = self.X[cand] @ q
        kk = min(k, len(cand))
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[np.argsort(-sims[top], kind="stable")]
        return ([(int(cand[i]), float(1 - sims[i])) for i in top],
                {"dist_computations": len(cand) + len(self.C), "candidates": len(cand)})
