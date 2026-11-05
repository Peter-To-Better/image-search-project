"""特徵擷取的統一介面：批次計算圖庫特徵並快取到 data/features/<name>.npz。"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from . import bovw as bovw_mod
from .color import color_hist
from .config import FEATURE_DIR
from .dataset import open_image
from .hashing import HASHERS

# kind: hash → 64-bit 整數，用 Hamming 距離
#       dense → L2 正規化的浮點向量，用餘弦距離
FEATURES = {
    "ahash":    {"kind": "hash",  "label": "aHash 平均雜湊",       "family": "感知雜湊"},
    "dhash":    {"kind": "hash",  "label": "dHash 差異雜湊",       "family": "感知雜湊"},
    "phash":    {"kind": "hash",  "label": "pHash 感知雜湊 (DCT)", "family": "感知雜湊"},
    "color":    {"kind": "dense", "label": "HSV 顏色直方圖",       "family": "全域手工特徵"},
    "bovw":     {"kind": "dense", "label": "SIFT + 視覺詞袋",      "family": "局部特徵"},
    "resnet50": {"kind": "dense", "label": "ResNet-50 (CNN)",      "family": "深度學習"},
    "dinov2":   {"kind": "dense", "label": "DINOv2 (自監督 ViT)",  "family": "深度學習"},
    "clip":     {"kind": "dense", "label": "CLIP (圖文多模態)",    "family": "深度學習"},
}
DEEP = {"resnet50", "dinov2", "clip"}
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)


# ---------- 單張圖片 (worker 用，需為模組層級函式) ----------
def _hash_one(args):
    name, path = args
    return HASHERS[name](open_image(path))


def _color_one(path):
    return color_hist(open_image(path))


def _sift_one(path):
    return bovw_mod.sift_descriptors(open_image(path))


def _pmap(fn, items, desc):
    if N_WORKERS == 1 or len(items) < 64:
        return [fn(x) for x in tqdm(items, desc=desc)]
    with ProcessPoolExecutor(N_WORKERS) as ex:
        return list(tqdm(ex.map(fn, items, chunksize=32), total=len(items), desc=desc))


# ---------- 深度模型 (延遲載入，整個程式只載一次) ----------
_deep_cache = {}


def get_deep(name):
    if name not in _deep_cache:
        from .deep import DEEP_MODELS
        _deep_cache[name] = DEEP_MODELS[name]()
    return _deep_cache[name]


def _deep_batch(name, paths, bs=32):
    m = get_deep(name)
    out = []
    for i in tqdm(range(0, len(paths), bs), desc=name):
        out.append(m.embed_images([open_image(p) for p in paths[i:i + bs]]))
    return np.concatenate(out).astype(np.float32)


# ---------- 對外介面 ----------
def compute(name, paths):
    """計算一批圖片的特徵 (圖庫用)。回傳 dict，至少含 'X'。"""
    t0 = time.time()
    if name in HASHERS:
        X = np.array(_pmap(_hash_one, [(name, p) for p in paths], name), dtype=np.uint64)
        out = {"X": X}
    elif name == "color":
        out = {"X": np.stack(_pmap(_color_one, paths, name)).astype(np.float32)}
    elif name == "bovw":
        descs = _pmap(_sift_one, paths, "SIFT")
        print(f"  平均每張 {np.mean([len(d) for d in descs]):.0f} 個 SIFT 描述子，開始 k-means…")
        model = bovw_mod.BoVW(k=1000).fit(descs)
        X = np.stack([model.encode(d) for d in tqdm(descs, desc="BoVW encode")])
        out = {"X": X.astype(np.float32), **model.state()}
    elif name in DEEP:
        out = {"X": _deep_batch(name, paths)}
    else:
        raise KeyError(name)
    out["seconds"] = np.float64(time.time() - t0)
    return out


def save(name, data):
    np.savez(FEATURE_DIR / f"{name}.npz", **data)


def load(name):
    f = FEATURE_DIR / f"{name}.npz"
    if not f.exists():
        return None
    return dict(np.load(f))


def available():
    return [n for n in FEATURES if (FEATURE_DIR / f"{n}.npz").exists()]


class QueryEncoder:
    """把單張 (或一批) 查詢圖轉成特徵，與圖庫特徵在同一個空間。"""

    def __init__(self, name, stored=None):
        self.name = name
        if name == "bovw":
            stored = stored if stored is not None else load("bovw")
            self.bovw = bovw_mod.BoVW.from_state(stored["centers"], stored["idf"])

    def __call__(self, img):
        return self.batch([img])[0]

    def batch(self, imgs):
        n = self.name
        if n in HASHERS:
            return [HASHERS[n](i) for i in imgs]
        if n == "color":
            return np.stack([color_hist(i) for i in imgs])
        if n == "bovw":
            return np.stack([self.bovw.encode(bovw_mod.sift_descriptors(i)) for i in imgs])
        return get_deep(n).embed_images(imgs)
