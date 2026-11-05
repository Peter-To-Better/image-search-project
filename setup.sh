#!/usr/bin/env bash
# 一次安裝所有套件（macOS / Linux）
set -e
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 9), "需要 Python 3.9 以上"' || exit 1
echo "▶ 使用 $($PY --version)"

if [ ! -d .venv ]; then $PY -m venv .venv; fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
if [ "$1" == "--deep" ]; then pip install -r requirements-deep.txt; fi

if ! command -v npm >/dev/null; then
  echo "⚠ 找不到 npm，請先安裝 Node.js（brew install node 或 https://nodejs.org）"; exit 1
fi
(cd frontend && npm install && npm run build)
echo "✅ 安裝完成。執行 ./start.sh 啟動 Demo"
