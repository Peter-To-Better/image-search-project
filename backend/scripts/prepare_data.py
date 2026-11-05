"""步驟 1：下載 Caltech-101 並切分「圖庫 / 查詢集」，同時產生加工過的近似副本查詢圖。

用法：python scripts/prepare_data.py
      python scripts/prepare_data.py --dataset /path/to/my_images   # 改用自己的資料集 (每個子資料夾一類)
"""
import argparse
import shutil
import tarfile
import urllib.request
import zipfile

import _path  # noqa: F401
from imgsearch.config import DATASET_DIR, RAW
from imgsearch.dataset import make_split

URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/caltech101.zip",
    "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip",
]


_last = [-1]


def _progress(block, bs, total):
    if total > 0:
        pct = min(100, block * bs * 100 // total)
        if pct != _last[0]:
            _last[0] = pct
            print(f"\r  {pct:3d}%", end="", flush=True)


def _find_class_root(base):
    for p in base.rglob("airplanes"):
        if p.is_dir():
            return p.parent
    return None


def download():
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"資料集已存在：{DATASET_DIR}")
        return
    tmp = RAW / "_download"
    tmp.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        try:
            z = tmp / "caltech101.zip"
            print(f"下載 {url}")
            urllib.request.urlretrieve(url, z, _progress)
            print("\n解壓縮中…")
            with zipfile.ZipFile(z) as f:
                f.extractall(tmp)
            for tgz in tmp.rglob("101_ObjectCategories.tar.gz"):     # 官方版是 zip 裡包 tar.gz
                with tarfile.open(tgz) as t:
                    t.extractall(tmp)
            root = _find_class_root(tmp)
            if root is None:
                raise RuntimeError("解壓後找不到類別資料夾")
            shutil.move(str(root), str(DATASET_DIR))
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"完成 → {DATASET_DIR}")
            return
        except Exception as e:  # noqa: BLE001
            print(f"\n  失敗：{e}，嘗試下一個來源…")
    raise SystemExit("所有下載來源都失敗，請手動下載 Caltech-101 解壓到 data/raw/caltech101/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="自訂資料集資料夾 (每個子資料夾 = 一個類別)")
    a = ap.parse_args()
    if a.dataset:
        from pathlib import Path
        split = make_split(Path(a.dataset))
    else:
        download()
        split = make_split()
    print(f"類別數 {len(split['classes'])}｜圖庫 {len(split['gallery'])} 張｜"
          f"語意查詢 {len(split['semantic_queries'])} 張｜近似副本查詢 {len(split['duplicate_queries'])} 張")
