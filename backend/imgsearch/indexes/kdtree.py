"""KD-tree (Bentley, 1975)：精確的 k-NN 搜尋。

建樹：每層選「變異數最大」的維度，以中位數切成左右兩半，直到葉節點 ≤ leaf_size。
      時間 O(N log N)。
查詢：先往查詢點所在的葉子走，再回溯；若「查詢點到切割超平面的距離」大於
      目前第 k 近的距離，另一側整棵子樹可以剪掉。

低維 (d ≲ 20) 時查詢接近 O(log N)；
高維時幾乎每個超平面都離查詢點很近 → 剪不掉 → 退化成 O(N) (維度災難)。
這個現象可以用 scripts/evaluate.py 的 kd 實驗觀察。
"""
import heapq

import numpy as np


class KDTree:
    name = "kdtree"
    exact = True

    def __init__(self, leaf_size=16):
        self.leaf_size = leaf_size

    def build(self, X):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        # 節點以平行陣列存放：dim, split, left, right, (leaf 時) start, end
        self.dim, self.split, self.left, self.right, self.lo, self.hi = [], [], [], [], [], []
        self.perm = np.arange(len(X))
        self._build(0, len(X))
        self.perm_X = self.X[self.perm]
        return self

    def _new(self):
        for a in (self.dim, self.split, self.left, self.right, self.lo, self.hi):
            a.append(-1)
        return len(self.dim) - 1

    def _build(self, lo, hi):
        nid = self._new()
        self.lo[nid], self.hi[nid] = lo, hi
        if hi - lo <= self.leaf_size:
            return nid
        idx = self.perm[lo:hi]
        pts = self.X[idx]
        dim = int(np.argmax(pts.var(axis=0)))
        mid = (hi - lo) // 2
        order = np.argpartition(pts[:, dim], mid)
        self.perm[lo:hi] = idx[order]
        self.dim[nid] = dim
        self.split[nid] = float(self.X[self.perm[lo + mid], dim])
        self.left[nid] = self._build(lo, lo + mid)
        self.right[nid] = self._build(lo + mid, hi)
        return nid

    def search(self, q, k=10):
        q = np.asarray(q, np.float32)
        best = []                     # max-heap (-dist², id)
        n_dist = 0

        def visit(nid):
            nonlocal n_dist
            if self.dim[nid] == -1:          # 葉節點：逐一計算距離
                lo, hi = self.lo[nid], self.hi[nid]
                d2 = ((self.perm_X[lo:hi] - q) ** 2).sum(1)
                n_dist += hi - lo
                for j in range(hi - lo):
                    item = (-float(d2[j]), int(self.perm[lo + j]))
                    if len(best) < k:
                        heapq.heappush(best, item)
                    elif item[0] > best[0][0]:
                        heapq.heapreplace(best, item)
                return
            diff = float(q[self.dim[nid]]) - self.split[nid]
            near, far = (self.left[nid], self.right[nid]) if diff < 0 else (self.right[nid], self.left[nid])
            visit(near)
            # 剪枝：超平面距離² ≥ 第 k 近距離² → 另一側不可能有更近的點
            if len(best) < k or diff * diff < -best[0][0]:
                visit(far)

        visit(0)
        res = sorted(((i, -nd) for nd, i in best), key=lambda t: t[1])
        # 向量已 L2 正規化：‖a-b‖² = 2 - 2cos → 餘弦距離 = ‖a-b‖²/2
        return [(i, d2 / 2) for i, d2 in res], {"dist_computations": n_dist}
