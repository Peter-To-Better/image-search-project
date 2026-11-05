#!/usr/bin/env bash
# 啟動 Demo：http://localhost:8000
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
if [ ! -f data/split.json ]; then
  echo "⚠ 還沒有資料，請先執行 ./run_experiments.sh"; exit 1
fi
if [ ! -d frontend/dist ]; then (cd frontend && npm run build); fi
( sleep 3; open "http://localhost:8000" 2>/dev/null || xdg-open "http://localhost:8000" 2>/dev/null || true ) &
cd backend
python -m uvicorn server:app --port 8000
