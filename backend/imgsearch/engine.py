"""搜尋引擎：把「特徵」與「索引結構」組合起來。"""
import threading
import time

from . import features as F
from .dataset import load_split
from .indexes import (HNSW, IVF, BKTree, BruteForceCosine, BruteForceHamming, InvertedIndex,
                      KDTree, RandomHyperplaneLSH, faiss_available)

INDEXES = {
    "brute":    {"label": "暴力線性掃描",          "exact": True,  "complexity": "O(N·d)"},
    "bktree":   {"label": "BK-tree (度量樹)",      "exact": True,  "complexity": "平均 ≈ O(log N)"},
    "kdtree":   {"label": "KD-tree",               "exact": True,  "complexity": "低維 O(log N)，高維退化 O(N)"},
    "lsh":      {"label": "LSH 隨機超平面",        "exact": False, "complexity": "次線性 O(N^ρ)"},
    "ivf":      {"label": "IVF 分群倒排",          "exact": False, "complexity": "≈ O(√N·d)"},
    "inverted": {"label": "倒排索引 (TF-IDF)",     "exact": True,  "complexity": "O(Σ posting 長度)"},
    "hnsw":     {"label": "HNSW 圖索引 (FAISS)",   "exact": False, "complexity": "≈ O(log N)"},
}


def index_options(feat):
    if F.FEATURES[feat]["kind"] == "hash":
        return ["brute", "bktree"]
    opts = ["brute", "kdtree", "lsh", "ivf"]
    if feat == "bovw":
        opts.insert(1, "inverted")
    if faiss_available():
        opts.append("hnsw")
    return opts


def make_index(feat, name, X, **kw):
    kind = F.FEATURES[feat]["kind"]
    if kind == "hash":
        return {"brute": BruteForceHamming, "bktree": BKTree}[name]().build(X)
    if name == "brute":
        return BruteForceCosine().build(X)
    if name == "kdtree":
        return KDTree(**kw).build(X)
    if name == "lsh":
        defaults = {"n_bits": 10, "n_tables": 16, "multiprobe": 1}   # 依 LSH 實驗挑選
        return RandomHyperplaneLSH(**{**defaults, **kw}).build(X)
    if name == "ivf":
        return IVF(**kw).build(X)
    if name == "inverted":
        return InvertedIndex().build(X)
    if name == "hnsw":
        return HNSW(**kw).build(X)
    raise KeyError(name)


class Engine:
    def __init__(self):
        self.split = load_split()
        self.gallery = self.split["gallery"]
        self.data = {f: F.load(f) for f in F.available()}
        self.encoders, self.indexes = {}, {}
        self.lock = threading.Lock()

    def features(self):
        return list(self.data)

    def encoder(self, feat):
        if feat not in self.encoders:
            self.encoders[feat] = F.QueryEncoder(feat, self.data.get(feat))
        return self.encoders[feat]

    def index(self, feat, idx):
        key = (feat, idx)
        with self.lock:
            if key not in self.indexes:
                t0 = time.perf_counter()
                self.indexes[key] = make_index(feat, idx, self.data[feat]["X"])
                print(f"建立索引 {feat}/{idx}：{time.perf_counter() - t0:.2f}s")
            return self.indexes[key]

    def warmup(self):
        """背景預先建立所有索引，避免第一次查詢時才建索引。"""
        for f in self.features():
            self.encoder(f)
            for i in index_options(f):
                self.index(f, i)

    def search(self, feat, idx, img, k=12):
        tb = time.perf_counter()
        index = self.index(feat, idx)                    # 第一次會建索引 (不計入查詢時間)
        t0 = time.perf_counter()
        q = self.encoder(feat)(img)
        t1 = time.perf_counter()
        res, stats = index.search(q, k)
        t2 = time.perf_counter()
        return {
            "feature": feat, "index": idx, "build_ms": (t0 - tb) * 1000,
            "encode_ms": (t1 - t0) * 1000, "search_ms": (t2 - t1) * 1000,
            "dist_computations": stats.get("dist_computations"),
            "gallery_size": len(self.gallery),
            "results": [{"id": i, "distance": d, **self.gallery[i]} for i, d in res],
        }

    def search_text(self, text, idx="brute", k=12):
        t0 = time.perf_counter()
        q = F.get_deep("clip").embed_text([text])[0]
        t1 = time.perf_counter()
        res, stats = self.index("clip", idx).search(q, k)
        t2 = time.perf_counter()
        return {"feature": "clip", "index": idx, "encode_ms": (t1 - t0) * 1000,
                "search_ms": (t2 - t1) * 1000, "dist_computations": stats.get("dist_computations"),
                "gallery_size": len(self.gallery),
                "results": [{"id": i, "distance": d, **self.gallery[i]} for i, d in res]}
