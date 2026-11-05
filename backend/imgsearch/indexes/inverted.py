"""倒排索引 (Inverted Index)：給 BoVW 的稀疏 TF-IDF 向量使用。

視覺單字 w → [(含有 w 的圖片 id, 權重), ...]
查詢時只需走訪「查詢圖出現過的單字」的 posting list，
沒有共同單字的圖完全不用碰 → 與文字搜尋引擎 (Google、Lucene) 相同的原理。

時間：O(Σ_w∈q |posting(w)|)，通常遠小於 O(N·K)
"""
import numpy as np


class InvertedIndex:
    name = "inverted"
    exact = True

    def build(self, X):
        X = np.asarray(X, np.float32)
        self.N = len(X)
        self.postings = []
        for w in range(X.shape[1]):
            ids = np.nonzero(X[:, w])[0].astype(np.int32)
            self.postings.append((ids, X[ids, w].copy()))
        return self

    def search(self, q, k=10):
        scores = np.zeros(self.N, np.float32)
        touched = 0
        for w in np.nonzero(q)[0]:
            ids, wt = self.postings[w]
            scores[ids] += q[w] * wt
            touched += len(ids)
        kk = min(k, self.N)
        top = np.argpartition(-scores, kk - 1)[:kk]
        top = top[np.argsort(-scores[top], kind="stable")]
        n_cand = int((scores > 0).sum())        # 至少共享一個視覺單字的圖片數
        return ([(int(i), float(1 - scores[i])) for i in top],
                {"dist_computations": n_cand, "candidates": n_cand, "postings_touched": touched})

    def avg_posting_len(self):
        return float(np.mean([len(p[0]) for p in self.postings]))
