#!/usr/bin/env bash
# 下載資料集與模型，並計算特徵（不跑實驗）。跑完就能 ./start.sh
#
# 資料來源：
#   Caltech-101  https://data.caltech.edu/records/mzrjq-6wc02
#                （備用鏡像 https://github.com/ultralytics/assets/releases/download/v0.0.0/caltech101.zip）
#   ResNet-50    https://github.com/onnx/models（resnet50-v2-7.onnx，算特徵時自動下載）
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then echo "⚠ 請先執行 ./setup.sh"; exit 1; fi
source .venv/bin/activate
cd backend
python scripts/prepare_data.py        # 下載 Caltech-101 → data/raw/，切分並產生變形查詢圖
python scripts/build_features.py "$@" # 下載 ResNet-50 → data/models/，計算特徵 → data/features/
echo "✅ 資料準備完成。執行 ./start.sh 啟動 Demo"
