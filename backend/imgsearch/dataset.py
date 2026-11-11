"""資料集載入與「圖庫 / 查詢集」切分。

Caltech-101：101 類物體 + 1 個背景類，共約 9,000 張。
 - 排除 BACKGROUND_Google (雜訊類別，沒有語意)
 - 排除 Faces_easy (與 Faces 是同一批照片的不同裁切，會造成評估錯誤)

兩種查詢：
 1. 語意查詢 (semantic)：每類保留 QUERIES_PER_CLASS 張「不在圖庫中」的圖，
    正確答案 = 圖庫中同類別的所有圖。
 2. 近似副本查詢 (duplicate)：從圖庫抽 DUP_SOURCE_COUNT 張，套用各種變形，
    正確答案 = 變形前的那一張原圖。
"""
import json
import random
import zlib
from pathlib import Path

from PIL import Image

from .config import (DATASET_DIR, DUP_QUERY_DIR, DUP_SOURCE_COUNT, QUERIES_PER_CLASS, ROOT,
                     SEED, SPLIT_FILE)
from .transforms import TRANSFORMS

EXCLUDE = {"BACKGROUND_Google", "Faces_easy"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def make_split(dataset_dir: Path = DATASET_DIR, queries_per_class=QUERIES_PER_CLASS,
               dup_sources=DUP_SOURCE_COUNT, seed=SEED):
    rng = random.Random(seed)
    classes = sorted(d.name for d in dataset_dir.iterdir() if d.is_dir() and d.name not in EXCLUDE)
    gallery, sem_queries = [], []
    for c in classes:
        files = sorted(f for f in (dataset_dir / c).iterdir() if f.suffix.lower() in EXTS)
        rng.shuffle(files)
        nq = queries_per_class if len(files) > queries_per_class * 3 else 0
        sem_queries += [{"path": rel(f), "label": c} for f in files[:nq]]
        gallery += [{"path": rel(f), "label": c} for f in files[nq:]]

    # 產生近似副本查詢圖
    DUP_QUERY_DIR.mkdir(parents=True, exist_ok=True)
    src_ids = rng.sample(range(len(gallery)), min(dup_sources, len(gallery)))
    dup_queries = []
    for gid in src_ids:
        img = Image.open(ROOT / gallery[gid]["path"]).convert("RGB")
        for tname, fn in TRANSFORMS.items():
            out = DUP_QUERY_DIR / f"{gid:05d}_{tname}.jpg"
            fn(img, random.Random(gid * 1009 + zlib.crc32(tname.encode()))).convert("RGB").save(out, quality=92)
            dup_queries.append({"path": rel(out), "source_id": gid, "transform": tname,
                                "label": gallery[gid]["label"]})

    split = {"classes": classes, "gallery": gallery, "semantic_queries": sem_queries,
             "duplicate_queries": dup_queries}
    SPLIT_FILE.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    return split


def load_split():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError("找不到 data/split.json，請先執行 python scripts/prepare_data.py")
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


def open_image(path) -> Image.Image:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return Image.open(p).convert("RGB")
