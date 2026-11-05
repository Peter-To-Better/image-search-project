"""FastAPI 後端：提供搜尋 API 給 React 前端。

啟動：uvicorn server:app --reload --port 8000   (在 backend/ 資料夾內)
"""
import base64
import io
import json
import random
import threading
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from imgsearch import features as F
from imgsearch.config import DATA, RESULTS_DIR, ROOT
from imgsearch.dataset import open_image
from imgsearch.engine import INDEXES, Engine, index_options
from imgsearch.transforms import TRANSFORM_LABELS, TRANSFORMS

app = FastAPI(title="以圖搜圖 Demo API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/data", StaticFiles(directory=DATA), name="data")

engine = Engine()
print(f"圖庫 {len(engine.gallery)} 張，可用特徵：{engine.features()}")
threading.Thread(target=engine.warmup, daemon=True).start()


def _thumb_data_url(img: Image.Image) -> str:
    im = img.copy()
    im.thumbnail((400, 400))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def _resolve_query(file, gallery_id, query_path, transform, seed):
    """查詢圖可以是：上傳的檔案 / 圖庫中的某張圖 / 查詢集的某張圖，並可再套用變形。"""
    truth = {}
    if file is not None:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    elif gallery_id is not None:
        if not 0 <= gallery_id < len(engine.gallery):
            raise HTTPException(400, "gallery_id 超出範圍")
        g = engine.gallery[gallery_id]
        img = open_image(g["path"])
        truth = {"source_id": gallery_id, "label": g["label"]}
    elif query_path:
        q = next((q for q in engine.split["semantic_queries"] + engine.split["duplicate_queries"]
                  if q["path"] == query_path), None)
        if q is None:
            raise HTTPException(400, "找不到此查詢圖")
        img = open_image(q["path"])
        truth = {"label": q["label"], **({"source_id": q["source_id"]} if "source_id" in q else {})}
    else:
        raise HTTPException(400, "請提供 file、gallery_id 或 query_path")
    if transform and transform != "none":
        if transform not in TRANSFORMS:
            raise HTTPException(400, f"未知的變形 {transform}")
        img = TRANSFORMS[transform](img, random.Random(seed))
    return img, truth


@app.get("/api/info")
def info():
    feats = engine.features()
    return {
        "gallery_size": len(engine.gallery),
        "classes": engine.split["classes"],
        "features": [{"name": f, **F.FEATURES[f], "indexes": index_options(f)} for f in feats],
        "indexes": INDEXES,
        "transforms": [{"name": "none", "label": "不變形"}] +
                      [{"name": k, "label": v} for k, v in TRANSFORM_LABELS.items()],
        "text_search": "clip" in feats,
    }


@app.get("/api/samples")
def samples(n: int = 12, kind: str = "semantic", seed: Optional[int] = None):
    """隨機取幾張示範用的查詢圖。kind = semantic (不在圖庫中) | gallery。"""
    r = random.Random(seed)
    if kind == "gallery":
        ids = r.sample(range(len(engine.gallery)), n)
        return [{"gallery_id": i, **engine.gallery[i]} for i in ids]
    qs = engine.split["semantic_queries"]
    return r.sample(qs, min(n, len(qs)))


@app.post("/api/search")
async def search(feature: str = Form(...), index: str = Form("brute"), k: int = Form(12),
                 file: Optional[UploadFile] = File(None), gallery_id: Optional[int] = Form(None),
                 query_path: Optional[str] = Form(None), transform: Optional[str] = Form(None),
                 seed: int = Form(0)):
    if feature not in engine.features():
        raise HTTPException(400, f"特徵 {feature} 尚未建立")
    if index not in index_options(feature):
        raise HTTPException(400, f"{feature} 不支援索引 {index}")
    img, truth = await _resolve_query(file, gallery_id, query_path, transform, seed)
    out = engine.search(feature, index, img, k)
    return {**out, "query_image": _thumb_data_url(img), "truth": truth}


@app.post("/api/compare")
async def compare(k: int = Form(8), file: Optional[UploadFile] = File(None),
                  gallery_id: Optional[int] = Form(None), query_path: Optional[str] = Form(None),
                  transform: Optional[str] = Form(None), seed: int = Form(0)):
    """同一張查詢圖，跑所有特徵 (都用精確的暴力搜尋) 方便並排比較。"""
    img, truth = await _resolve_query(file, gallery_id, query_path, transform, seed)
    runs = [engine.search(f, "brute", img, k) for f in engine.features()]
    for r in runs:
        r["label"] = F.FEATURES[r["feature"]]["label"]
    return {"query_image": _thumb_data_url(img), "truth": truth, "runs": runs}


@app.post("/api/text")
def text_search(text: str = Form(...), index: str = Form("brute"), k: int = Form(12)):
    if "clip" not in engine.features():
        raise HTTPException(400, "需要先建立 CLIP 特徵才能以文字搜圖")
    return engine.search_text(text, index, k)


@app.get("/api/results")
def results():
    f = RESULTS_DIR / "results.json"
    if not f.exists():
        raise HTTPException(404, "尚未執行 scripts/evaluate.py")
    return json.loads(f.read_text())


# 若前端已 build (npm run build)，直接由後端提供網頁
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if f.is_file() else DIST / "index.html")
