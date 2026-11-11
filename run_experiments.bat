@echo off
chcp 65001 >nul
rem 從零開始：下載資料集、計算特徵、跑所有實驗（Windows）
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

if exist .venv\Scripts\activate.bat goto :ready
echo 還沒有安裝環境，需要先執行 setup.bat。
choice /c YN /m "現在幫你執行 setup.bat 嗎"
if errorlevel 2 goto :fail
call setup.bat
if errorlevel 1 goto :fail

:ready
call .venv\Scripts\activate.bat
cd backend
echo [1/3] 下載並切分資料集…
python scripts\prepare_data.py
if errorlevel 1 goto :fail
echo [2/3] 計算特徵…
python scripts\build_features.py %*
if errorlevel 1 goto :fail
echo [3/3] 跑實驗…
python scripts\evaluate.py
if errorlevel 1 goto :fail
echo.
echo 完成！結果在 results\，下一步：雙擊 start.bat 打開網頁
pause
exit /b 0

:fail
echo.
echo 執行沒有完成，請看上方的錯誤訊息（網路中斷的話直接重跑即可，已下載的會略過）。
pause
exit /b 1
