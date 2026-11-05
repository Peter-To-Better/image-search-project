"""HNSW (Hierarchical Navigable Small World, Malkov & Yashunin 2016)，使用 FAISS 實作。

多層的鄰近圖：上層稀疏 (長距離跳躍)、下層稠密 (精細搜尋)，
查詢時從最上層貪婪地往目標走，類似跳躍串列 (skip list) 的概念。
查詢約 O(log N)，是目前業界向量資料庫 (Milvus、Qdrant、pgvector) 的主流方法。
這裡當作「工業級」對照組，自己實作的是 LSH / IVF / KD-tree / BK-tree。
"""
import numpy as np


class HNSW:
    name = "hnsw"
    exact = False

    def __init__(self, M=32, ef_search=64, ef_construction=200):
        self.M, self.efs, self.efc = M, ef_search, ef_construction

    def build(self, X):
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index = faiss.IndexHNSWFlat(X.shape[1], self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.efc
        self.index.add(X)
        self.index.hnsw.efSearch = self.efs
        return self

    def search(self, q, k=10):
        D, I = self.index.search(q[None, :].astype(np.float32), k)
        return ([(int(i), float(1 - d)) for i, d in zip(I[0], D[0]) if i >= 0],
                {"dist_computations": None})


def faiss_available():
    try:
        import faiss  # noqa
        return True
    except ImportError:
        return False
