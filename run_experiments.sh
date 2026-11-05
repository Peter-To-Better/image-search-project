#!/usr/bin/env bash
# 從零開始：下載資料集 → 計算特徵 → 跑所有實驗
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
cd backend
python scripts/prepare_data.py
python scripts/build_features.py "$@"
python scripts/evaluate.py
echo "✅ 完成！結果在 results/，執行 ./start.sh 打開網頁"
