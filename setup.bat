@echo off
chcp 65001 >nul
rem ============================================================
rem  以圖搜圖專題 - Windows 一鍵安裝
rem  用法：setup.bat [--deep]
rem  1. 檢查 Python 3.10+（64 位元）、Node.js 18+、VC++ 執行階段
rem  2. 缺什麼就詢問後自動安裝（優先用 winget，沒有就從官網下載）
rem  3. 建立 .venv、安裝 Python 套件、npm install、build 前端
rem ============================================================
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set "DEEP="
if /i "%~1"=="--deep" set "DEEP=1"

set "PY_URL=https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe"
set "NODE_URL=https://nodejs.org/dist/v20.18.1/node-v20.18.1-x64.msi"
set "VC_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe"

echo.
echo ============================================================
echo   以圖搜圖專題 - Windows 安裝程式
echo ============================================================
echo.

if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
  echo [注意] 這台是 ARM 版 Windows，部分套件可能裝不起來；建議改用 x64 電腦。
  echo.
)

set "HAS_WINGET="
where winget >nul 2>nul
if not errorlevel 1 set "HAS_WINGET=1"

rem ---------------- 步驟 1：檢查環境 ----------------
echo [1/4] 檢查環境…
call :detect_all
call :print_status
if not defined MISSING goto :env_ok

echo.
echo 以上標示「缺少」的項目需要安裝。
echo 安裝過程中如果跳出「是否允許此 App 變更您的裝置」，請按「是」。
choice /c YN /m "要自動幫你安裝嗎"
if errorlevel 2 goto :manual_help

if not defined PY_OK call :install_python
if not defined NODE_OK call :install_node
if not defined VC_OK call :install_vc

echo.
echo 重新檢查環境…
call :refresh_path
call :detect_all
call :print_status
if defined MISSING (
  echo.
  echo [提醒] 剛裝好的程式有時要開新視窗才抓得到。
  echo        請關掉這個視窗，再重新雙擊 setup.bat 一次。
  goto :fail
)

:env_ok
rem ---------------- 步驟 2：Python 虛擬環境與套件 ----------------
echo.
echo [2/4] 建立 Python 虛擬環境 .venv …
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo   舊的 .venv 壞掉了，重新建立…
    rmdir /s /q .venv
  )
)
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)
call .venv\Scripts\activate.bat

echo.
echo [3/4] 安裝 Python 套件（第一次約 3-5 分鐘）…
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [錯誤] Python 套件安裝失敗。常見原因：網路不穩，請重跑一次 setup.bat。
  goto :fail
)
if defined DEEP (
  python -m pip install -r requirements-deep.txt
  if errorlevel 1 goto :fail
)
python -c "import numpy, cv2, onnxruntime, faiss, fastapi" >nul 2>nul
if errorlevel 1 (
  echo [錯誤] 套件裝好了但無法載入，多半是缺 Microsoft Visual C++ 執行階段。
  echo        請安裝 %VC_URL% 後重跑 setup.bat。
  python -c "import numpy, cv2, onnxruntime, faiss, fastapi"
  goto :fail
)

rem ---------------- 步驟 3：前端 ----------------
echo.
echo [4/4] 安裝並 build 前端…
pushd frontend
call npm install
if errorlevel 1 (popd & goto :fail)
call npm run build
if errorlevel 1 (popd & goto :fail)
popd

echo.
echo ============================================================
echo   安裝完成！
echo   下一步：雙擊 run_experiments.bat（下載資料、跑實驗）
echo   之後：  雙擊 start.bat（打開網頁 Demo）
echo ============================================================
pause
exit /b 0

rem ============================================================
rem  子程序
rem ============================================================

:detect_all
set "MISSING="
call :detect_python
call :detect_node
call :detect_vc
if not defined PY_OK set "MISSING=1"
if not defined NODE_OK set "MISSING=1"
if not defined VC_OK set "MISSING=1"
exit /b 0

:detect_python
rem 找 64 位元的 Python 3.10 以上；優先 3.12 / 3.11 / 3.10（套件支援最完整）
set "PY="
set "PY_OK="
if defined PYTHON (
  set "PY=%PYTHON%"
  set "PY_OK=1"
  exit /b 0
)
for %%c in ("py -3.12" "py -3.11" "py -3.10" "py -3" "python") do (
  if not defined PY (
    %%~c -c "import sys, struct; sys.exit(0 if sys.version_info >= (3, 10) and struct.calcsize('P') == 8 else 1)" >nul 2>nul && set "PY=%%~c"
  )
)
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
if defined PY set "PY_OK=1"
exit /b 0

:detect_node
set "NODE_OK="
where node >nul 2>nul
if errorlevel 1 if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
where npm >nul 2>nul
if errorlevel 1 exit /b 0
node -e "process.exit(+process.versions.node.split('.')[0] >= 18 ? 0 : 1)" >nul 2>nul
if not errorlevel 1 set "NODE_OK=1"
exit /b 0

:detect_vc
set "VC_OK="
if exist "%SystemRoot%\System32\vcruntime140_1.dll" if exist "%SystemRoot%\System32\msvcp140.dll" set "VC_OK=1"
exit /b 0

:print_status
if defined PY_OK (
  echo   [OK]   Python：%PY%
) else (
  echo   [缺少] Python 3.10 以上（64 位元）
)
if defined NODE_OK (
  echo   [OK]   Node.js 18 以上
) else (
  echo   [缺少] Node.js 18 以上
)
if defined VC_OK (
  echo   [OK]   Microsoft Visual C++ 執行階段
) else (
  echo   [缺少] Microsoft Visual C++ 執行階段
)
exit /b 0

:install_python
echo.
echo ^>^> 安裝 Python 3.12 …
if defined HAS_WINGET (
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  if not errorlevel 1 exit /b 0
  echo   winget 安裝失敗，改從 python.org 下載安裝檔…
)
call :download "%PY_URL%" "%TEMP%\python-installer.exe"
if errorlevel 1 exit /b 1
start "" /wait "%TEMP%\python-installer.exe" /passive InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
exit /b 0

:install_node
echo.
echo ^>^> 安裝 Node.js LTS …
if defined HAS_WINGET (
  winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
  if not errorlevel 1 exit /b 0
  echo   winget 安裝失敗，改從 nodejs.org 下載安裝檔…
)
call :download "%NODE_URL%" "%TEMP%\node-installer.msi"
if errorlevel 1 exit /b 1
start "" /wait msiexec /i "%TEMP%\node-installer.msi" /passive
exit /b 0

:install_vc
echo.
echo ^>^> 安裝 Microsoft Visual C++ 執行階段 …
if defined HAS_WINGET (
  winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements
  if not errorlevel 1 exit /b 0
  echo   winget 安裝失敗，改從微軟官網下載…
)
call :download "%VC_URL%" "%TEMP%\vc_redist.x64.exe"
if errorlevel 1 exit /b 1
start "" /wait "%TEMP%\vc_redist.x64.exe" /install /passive /norestart
exit /b 0

:download
echo   下載 %~1
where curl >nul 2>nul
if not errorlevel 1 (
  curl -L --fail -o "%~2" "%~1"
  if not errorlevel 1 exit /b 0
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri '%~1' -OutFile '%~2'"
if errorlevel 1 (
  echo   [錯誤] 下載失敗，請確認網路連線。
  exit /b 1
)
exit /b 0

:refresh_path
rem 從登錄檔重新讀 PATH，讓剛裝好的程式在這個視窗就能用
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')"`) do set "PATH=%%p"
exit /b 0

:manual_help
echo.
echo 好的，請手動安裝以下軟體後，再重新執行 setup.bat：
if not defined PY_OK   echo   Python 3.12：https://www.python.org/downloads/  （安裝第一頁務必勾選 Add python.exe to PATH）
if not defined NODE_OK echo   Node.js LTS：https://nodejs.org
if not defined VC_OK   echo   VC++ 執行階段：%VC_URL%
echo 詳細步驟請看 WINDOWS.md
goto :fail

:fail
echo.
echo 安裝沒有完成，請看上方的訊息。需要協助時，把這個視窗截圖給組員。
pause
exit /b 1
