@echo off
chcp 65001 >nul
rem 啟動 Demo：http://localhost:8000（Windows）
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

if exist .venv\Scripts\activate.bat goto :check_data
echo 還沒有安裝環境，需要先執行 setup.bat。
choice /c YN /m "現在幫你執行 setup.bat 嗎"
if errorlevel 2 goto :fail
call setup.bat
if errorlevel 1 goto :fail

:check_data
if exist data\split.json goto :run
echo 還沒有資料集，需要先執行 run_experiments.bat（約 15-20 分鐘）。
choice /c YN /m "現在幫你執行 run_experiments.bat 嗎"
if errorlevel 2 goto :fail
call run_experiments.bat
if errorlevel 1 goto :fail

:run
call .venv\Scripts\activate.bat
if exist frontend\dist goto :serve
pushd frontend
call npm run build
if errorlevel 1 (popd & goto :fail)
popd

:serve
echo.
echo 啟動中… 瀏覽器會自動打開 http://localhost:8000
echo 要結束時，在這個視窗按 Ctrl+C，或直接關掉視窗。
rem 3 秒後自動打開瀏覽器
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://localhost:8000'"
cd backend
python -m uvicorn server:app --port 8000
exit /b %errorlevel%

:fail
pause
exit /b 1
