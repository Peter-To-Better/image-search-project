"""全域路徑與設定。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # image-search-project/
DATA = ROOT / "data"
RAW = DATA / "raw"                                   # 原始資料集
DATASET_DIR = RAW / "caltech101"                     # 每個子資料夾 = 一個類別
SPLIT_FILE = DATA / "split.json"                     # 圖庫 / 查詢集切分
DUP_QUERY_DIR = DATA / "queries_dup"                 # 自動產生的「加工查詢圖」
FEATURE_DIR = DATA / "features"                      # 特徵快取 (.npz)
MODEL_DIR = DATA / "models"                          # 下載的深度模型
RESULTS_DIR = ROOT / "results"                       # 實驗結果 (json + png)

SEED = 42
QUERIES_PER_CLASS = 3        # 語意查詢：每個類別保留幾張當查詢
DUP_SOURCE_COUNT = 150       # 近似副本查詢：從圖庫抽幾張來加工

for p in (DATA, RAW, FEATURE_DIR, MODEL_DIR, RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)
