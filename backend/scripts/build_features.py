"""步驟 2：計算圖庫所有圖片的特徵並快取。

用法：python scripts/build_features.py                       # 預設：ahash dhash phash color bovw resnet50
      python scripts/build_features.py --features dinov2 clip  # 需要先 pip install torch transformers
      python scripts/build_features.py --force                 # 重新計算
"""
import argparse

import _path  # noqa: F401
from imgsearch import features as F
from imgsearch.config import ROOT
from imgsearch.dataset import load_split

DEFAULT = ["ahash", "dhash", "phash", "color", "bovw", "resnet50"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=DEFAULT, choices=list(F.FEATURES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    paths = [str(ROOT / g["path"]) for g in load_split()["gallery"]]
    for name in a.features:
        if not a.force and F.load(name) is not None:
            print(f"[略過] {name} 已存在")
            continue
        print(f"=== {F.FEATURES[name]['label']} ({len(paths)} 張) ===")
        try:
            data = F.compute(name, paths)
        except RuntimeError as e:
            print(f"[錯誤] {name}: {e}")
            continue
        F.save(name, data)
        print(f"  完成，耗時 {float(data['seconds']):.1f}s，維度 {data['X'].shape}")
