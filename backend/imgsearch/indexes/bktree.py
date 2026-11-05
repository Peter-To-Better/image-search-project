"""BK-tree (Burkhard–Keller tree, 1973)：適用於任何「離散度量空間」的索引，這裡用 Hamming 距離。

結構：每個節點存一個元素，子節點以「與父節點的距離 d」作為邊的標籤。
剪枝原理 (三角不等式)：
    設查詢 q、節點 x、子樹 c 的邊標籤 = d(x, c 子樹內任一點 y) = e。
    |d(q, x) - d(x, y)| ≤ d(q, y)
    若要 d(q, y) ≤ τ，則必須 d(q,x) - τ ≤ e ≤ d(q,x) + τ ，
    其他子樹可以整個跳過。

建樹：平均 O(N log N)，最差 O(N²)
查詢：τ 小時接近 O(log N)；τ 越大剪枝越少，最差退化成 O(N)
k-NN 搜尋：τ 動態設為「目前第 k 好的距離」，找到更好的就縮小 τ。
"""
import heapq

from ..hashing import hamming


class BKTree:
    name = "bktree"
    exact = True

    def __init__(self, dist=hamming):
        self.dist = dist
        self.root = None       # 節點 = [item_id, value, {edge: child}]
        self.size = 0

    def add(self, item_id, value):
        self.size += 1
        if self.root is None:
            self.root = [item_id, value, {}]
            return
        node = self.root
        while True:
            d = self.dist(value, node[1])
            child = node[2].get(d)
            if child is None:
                node[2][d] = [item_id, value, {}]
                return
            node = child

    def build(self, values):
        for i, v in enumerate(values):
            self.add(i, int(v))
        return self

    def range_search(self, q, radius):
        """回傳所有 d(q, x) ≤ radius 的元素。"""
        out, stack, n_dist = [], [self.root], 0
        while stack:
            node = stack.pop()
            d = self.dist(q, node[1])
            n_dist += 1
            if d <= radius:
                out.append((node[0], d))
            for e, child in node[2].items():
                if d - radius <= e <= d + radius:
                    stack.append(child)
        out.sort(key=lambda t: (t[1], t[0]))
        return out, {"dist_computations": n_dist}

    def search(self, q, k=10):
        q = int(q)
        best = []                          # max-heap：(-dist, -id)
        stack, n_dist = [self.root], 0
        tau = float("inf")
        while stack:
            node = stack.pop()
            d = self.dist(q, node[1])
            n_dist += 1
            if len(best) < k:
                heapq.heappush(best, (-d, -node[0]))
            elif (d, node[0]) < (-best[0][0], -best[0][1]):
                heapq.heapreplace(best, (-d, -node[0]))
            if len(best) == k:
                tau = -best[0][0]
            # 先放遠的、後放近的 → 近的先被 pop，τ 會更快縮小
            for e in sorted(node[2], key=lambda e: -abs(e - d)):
                if d - tau <= e <= d + tau:
                    stack.append(node[2][e])
        res = sorted(((-i, -nd) for nd, i in best), key=lambda t: (t[1], t[0]))
        return [(int(i), float(dd)) for i, dd in res], {"dist_computations": n_dist}

    def depth(self):
        best, stack = 0, [(self.root, 1)]
        while stack:
            n, dep = stack.pop()
            best = max(best, dep)
            stack.extend((c, dep + 1) for c in n[2].values())
        return best
