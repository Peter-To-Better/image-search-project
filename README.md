# 以圖搜圖（Content-Based Image Retrieval）演算法專題

從「感知雜湊」到「深度向量檢索」，比較 **6 種特徵擷取方法 × 7 種索引結構**，並附 React 網頁 Demo。

- 自己實作的演算法：aHash / dHash / pHash（含 DCT）、HSV 直方圖、SIFT 視覺詞袋、**BK-tree、KD-tree、LSH（含 multi-probe）、IVF（含 k-means++）、倒排索引**
- 工業級對照組：FAISS HNSW
- 深度學習：ResNet-50（ONNX，免裝 PyTorch）、DINOv2 / CLIP（選用，需 torch）
- 資料集：Caltech-101（100 類，圖庫 7,942 張）

---

## 快速開始（macOS）

需要 **Python 3.10+** 與 **Node.js 18+**（沒有的話：`brew install python node`）。

```bash
cd ~/Desktop/image-search-project
./setup.sh            # 建立 .venv、安裝 Python 套件、npm install、build 前端（約 3–5 分鐘）
./start.sh            # 啟動 → 自動打開 http://localhost:8000
```

> 資料集、特徵、ResNet 模型和實驗結果都已經預先算好放在 `data/` 與 `results/`，裝完套件就能直接 Demo。

### 想從零重跑全部流程

```bash
rm -rf data results      # 清掉預先算好的東西（可選）
./run_experiments.sh     # 下載資料集 → 算特徵 → 跑實驗（約 15–20 分鐘）
```

### 加入 DINOv2 / CLIP（選用，可以用文字搜圖）

```bash
./setup.sh --deep
source .venv/bin/activate && cd backend
python scripts/build_features.py --features dinov2 clip
python scripts/evaluate.py
```

### 開發模式（改前端程式時用，會即時刷新）

```bash
# 終端機 1
source .venv/bin/activate && cd backend && python -m uvicorn server:app --reload --port 8000
# 終端機 2
cd frontend && npm run dev        # → http://localhost:5173
```

---

## 專案結構

```
image-search-project/
├── backend/
│   ├── imgsearch/
│   │   ├── hashing.py        # aHash / dHash / pHash（自己寫 DCT）＋ Hamming 距離
│   │   ├── color.py          # HSV 顏色直方圖（Hellinger 轉換）
│   │   ├── bovw.py           # SIFT → k-means 視覺單字 → TF-IDF
│   │   ├── deep.py           # ResNet-50 (ONNX)、DINOv2、CLIP
│   │   ├── features.py       # 特徵統一介面、批次計算與快取
│   │   ├── indexes/
│   │   │   ├── brute.py      # 暴力線性掃描（baseline）
│   │   │   ├── bktree.py     # BK-tree：三角不等式剪枝
│   │   │   ├── kdtree.py     # KD-tree：展示維度災難
│   │   │   ├── lsh.py        # 隨機超平面 LSH + multi-probe
│   │   │   ├── ivf.py        # IVF + 自己寫的 k-means++
│   │   │   ├── inverted.py   # 倒排索引（給 BoVW）
│   │   │   └── hnsw.py       # FAISS HNSW（對照組）
│   │   ├── dataset.py        # 圖庫 / 查詢集切分
│   │   ├── transforms.py     # 7 種影像變形（旋轉、裁切、亮度…）
│   │   ├── metrics.py        # mAP、P@k、Recall@k
│   │   └── engine.py         # 特徵 × 索引 組合成搜尋引擎
│   ├── scripts/
│   │   ├── prepare_data.py   # 步驟 1：下載資料集、切分、產生變形查詢圖
│   │   ├── build_features.py # 步驟 2：計算圖庫特徵
│   │   └── evaluate.py       # 步驟 3：7 個實驗 → results/results.json + 圖表
│   └── server.py             # FastAPI 後端
├── frontend/                 # React (Vite) Demo 網頁
├── data/                     # 資料集、特徵快取、模型
└── results/                  # 實驗結果 JSON 與 PNG 圖表（報告可以直接用）
```

---

## 實驗設計

| 查詢類型 | 數量 | 怎麼產生 | 正確答案 |
|---|---|---|---|
| 語意查詢 | 300 | 每類保留 3 張**不放進圖庫** | 圖庫中同類別的所有圖 |
| 近似副本查詢 | 1,050 | 從圖庫抽 150 張 × 7 種變形 | 變形前的原圖 |

7 個實驗（`backend/scripts/evaluate.py`）：

1. **準確度**：各特徵的語意 mAP、P@10、近似副本 Recall@1/@10
2. **穩健性**：每一種變形下找回原圖的比例
3. **特徵 × 索引**：查詢時間、距離計算比例、ANN Recall@10
4. **擴展性**：N = 1k → 64k 的查詢時間（log-log 圖，看斜率 = 複雜度）
5. **LSH 參數取捨**：K × L × multi-probe 的召回率與候選比例，加理論碰撞機率曲線
6. **KD-tree 維度災難**：PCA 降到 2–256 維時需要走訪的點比例
7. **BK-tree 剪枝**：搜尋半徑與走訪節點比例

## 實驗結果摘要（本機實測）

**實驗 1｜準確度**

| 特徵 | 語意 mAP | 語意 P@10 | 副本 R@1 | 副本 R@10 | 擷取時間/張 |
|---|---|---|---|---|---|
| aHash | 4.6% | 11.3% | 48.5% | 57.4% | 0.8 ms |
| dHash | 3.9% | 10.0% | 59.6% | 65.0% | 0.9 ms |
| pHash | 3.3% | 8.7% | 56.0% | 61.3% | 0.9 ms |
| HSV 直方圖 | 5.1% | 11.0% | 78.7% | 87.2% | 1.3 ms |
| SIFT + BoVW | 6.5% | 14.1% | 81.1% | 89.0% | 16.4 ms |
| **ResNet-50** | **65.7%** | **82.8%** | **87.0%** | **94.5%** | 38.9 ms |

（隨機猜的語意 mAP 約 1%。）

**實驗 2｜穩健性（Recall@1）**

| 特徵 | 旋轉 | 裁切 | 亮度 | JPEG | 縮小模糊 | 浮水印 | 翻轉 |
|---|---|---|---|---|---|---|---|
| pHash | 12% | 1% | 99% | 100% | 100% | 79% | 0% |
| HSV | 95% | 93% | **20%** | 73% | 91% | 79% | 100% |
| BoVW | 97% | 100% | 94% | 88% | **15%** | 100% | 74% |
| ResNet-50 | 97% | 93% | 100% | 91% | 31% | 100% | 97% |

**實驗 3｜索引（ResNet-50，N = 7,942）**

| 索引 | 查詢時間 | 距離計算比例 | Recall@10 |
|---|---|---|---|
| 暴力法 | 2.90 ms | 100% | 100% |
| KD-tree | 18.6 ms | 100%（剪不掉） | 100% |
| LSH (K=10, L=16, probe) | 2.15 ms | 17.8% | 90.4% |
| IVF (nprobe=8) | 0.91 ms | 10.5% | 98.0% |
| HNSW | 0.33 ms | — | 99.9% |

## 報告可以寫的重點（老師愛看的）

1. **沒有一種特徵通吃**：感知雜湊極快、極省空間，但只認得「幾乎同一張圖」，旋轉、裁切、翻轉就失效；CNN 有語意，卻怕圖縮得太小。這是**不變性（invariance）的取捨**。
2. **KD-tree 的維度災難**：2 維時只看 0.3% 的點，256 維時要看 96%，幾乎等於暴力法（實驗 6）。
3. **BK-tree 只適合小半徑**：半徑 ≤ 4 時只走訪不到 10% 的節點，但 k-NN 的第 10 名距離通常大於 16，剪枝失效（實驗 7）。這說明**資料結構的好壞取決於查詢的分布**。
4. **漸進複雜度和實際常數**：N = 8k 時 numpy 的暴力法（向量化）比純 Python 的 BK-tree 快上百倍，但從 log-log 圖的斜率可以看出，N 變大後次線性方法終究會贏（實驗 4）。
5. **LSH 理論與實測吻合**：碰撞機率 1 − (1 − p^K)^L 的 S 曲線解釋了為什麼 K 大會漏、L 大會慢；multi-probe 可以用更少的表達到相同的召回率（實驗 5）。
6. **實作細節的影響**：CNN 特徵全部 ≥ 0，擠在同一個象限，隨機超平面切不開。沒有**先減掉平均值（centering）**時，LSH 比暴力法還慢約 3 倍；置中後只需檢查 18% 的候選（見 `lsh.py`）。

## Demo 建議流程（5 分鐘）

1. **以圖搜圖** → 查詢集挑一張，用 ResNet-50 → 同類別的圖都找到了
2. 同一張切到 **pHash** → 結果幾乎都錯 → 「雜湊不懂語意」
3. **方法比較** → 圖庫原圖 + **旋轉** → 三種雜湊都找不到原圖，SIFT 和 CNN 找到
4. 同一張改成 **亮度** → 換成顏色直方圖失效、雜湊沒問題
5. **以圖搜圖** → ResNet-50，索引從暴力法切到 KD-tree / LSH / HNSW → 看「距離計算次數」的變化
6. 現場用手機拍一張照片上傳
7. **實驗結果** 頁講圖表

## 已知限制

- DINOv2 / CLIP 的程式碼使用標準的 Hugging Face API，但在建置環境中無法下載權重，**尚未實機跑過**；第一次執行請先確認 `build_features.py --features dinov2 clip` 能成功。
- N > 7,942 的擴展性實驗使用「真實向量 + 雜訊」合成的資料點。
- 查詢時間是 CPU 單執行緒實測，不同電腦數字會不同，但「距離計算比例」與機器無關。
