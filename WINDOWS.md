# Windows 安裝與執行教學（給完全沒裝過開發環境的同學）

只要照著做，**電腦上什麼都不用先裝**。缺的軟體（Python、Node.js、Visual C++ 執行階段）`setup.bat` 會先檢查，再問你要不要自動安裝。

> 需要：Windows 10（1809 以後）或 Windows 11、64 位元、能上網、約 3 GB 硬碟空間。
> 全部跑完大約 30 分鐘（大多是在下載和計算，不用一直盯著）。

---

## 步驟 0：下載專題

沒有 git 也沒關係：

1. 打開 https://github.com/Peter-To-Better/image-search-project
2. 按綠色的 **Code** 按鈕 → **Download ZIP**
3. 對下載的 ZIP 按右鍵 → **解壓縮全部**
4. 把資料夾放在**路徑簡單的位置**，例如 `C:\image-search-project`
   （避免放在 OneDrive 同步資料夾裡，同步會讓安裝變得很慢）

> 有 git 的話：`git clone https://github.com/Peter-To-Better/image-search-project.git`

## 步驟 1：雙擊 `setup.bat`（安裝環境，只要做一次）

1. 打開專題資料夾，雙擊 **`setup.bat`**
2. 如果跳出藍色視窗「Windows 已保護您的電腦」：按 **其他資訊** → **仍要執行**
3. 它會先列出檢查結果，例如：

   ```
   [1/4] 檢查環境…
     [缺少] Python 3.10 以上（64 位元）
     [缺少] Node.js 18 以上
     [OK]   Microsoft Visual C++ 執行階段
   要自動幫你安裝嗎 [Y,N]?
   ```

4. 按 **Y**，它會自動安裝缺少的軟體
   - 中途跳出「**是否允許此 App 變更您的裝置？**」→ 按 **是**
   - 安裝視窗會自己跑完、自己關掉，不用按任何東西
5. 接著它會自動安裝 Python 套件和建置網頁（約 3–5 分鐘）
6. 看到 **「安裝完成！」** 就可以了，按任意鍵關閉

> **如果看到「剛裝好的程式有時要開新視窗才抓得到」**：這是正常的。關掉視窗，再雙擊一次 `setup.bat` 就好，已經裝好的不會重裝。

## 步驟 2：雙擊 `run_experiments.bat`（下載資料、跑實驗，只要做一次）

- 會自動下載 Caltech-101 資料集（約 130 MB）與 ResNet-50 模型，然後計算特徵、跑 7 個實驗
- 大約 15–20 分鐘，畫面上會有進度條
- 看到「**完成！**」就可以了
- 網路斷掉的話直接重跑，已經下載的會略過

> 專題附了一份實驗結果（`results/` 資料夾）。不過網頁 Demo 需要圖庫和特徵，所以這一步還是要跑。

## 步驟 3：雙擊 `start.bat`（打開網頁 Demo，之後每次都用這個）

- 約 3 秒後瀏覽器會自動打開 http://localhost:8000
- 沒有自動打開的話，自己在瀏覽器輸入這個網址
- 第一次跳出 **Windows 防火牆** 詢問時，按「允許」或直接關掉都可以（只有自己電腦用得到）
- **用完要關閉**：回到黑色視窗按 `Ctrl + C`，或直接關掉視窗

> 小提醒：直接雙擊 `start.bat` 也行，它發現還沒安裝或還沒有資料時，會問你要不要先幫你跑步驟 1、2。

---

## 常見問題

| 狀況 | 解法 |
|---|---|
| 雙擊後視窗一閃就消失 | 在資料夾空白處按右鍵 →「在終端機中開啟」，輸入 `.\setup.bat` 執行，就看得到錯誤訊息 |
| 「Windows 已保護您的電腦」 | 按 **其他資訊** → **仍要執行** |
| 一直顯示 `[缺少] Python` | 關掉視窗重開 `setup.bat`；還是不行的話，到 https://www.python.org/downloads/ 手動安裝，安裝第一頁**一定要勾 `Add python.exe to PATH`** |
| 一直顯示 `[缺少] Node.js` | 到 https://nodejs.org 下載 LTS 版安裝，裝完重開 `setup.bat` |
| 「套件裝好了但無法載入」 | 安裝 [Visual C++ 執行階段](https://aka.ms/vs/17/release/vc_redist.x64.exe)，再重跑 `setup.bat` |
| `pip install` 失敗 / 下載中斷 | 通常是網路不穩，換個網路重跑即可 |
| 學校電腦不能安裝軟體（沒有系統管理員權限） | Python 會裝在使用者目錄，不需要權限；Node.js 和 VC++ 需要，請改用自己的電腦 |
| 網頁打不開 / 顯示 port 8000 被占用 | 確認沒有開兩個 `start.bat`；或重新開機後再試 |
| 中文字變成亂碼 | 不影響執行，可以忽略 |

還是卡住的話，把**整個黑色視窗截圖**傳給組員。

---

## 進階：加入 DINOv2 / CLIP（選用，可以用文字搜圖）

會額外下載 PyTorch（約 2 GB）。在專題資料夾開終端機：

```bat
setup.bat --deep
run_experiments.bat --features dinov2 clip
```

（在 PowerShell 裡要寫成 `.\setup.bat --deep`。）

## 進階：手動執行 Python 指令

```bat
.venv\Scripts\activate
cd backend
python scripts\evaluate.py
```

## 這些腳本做了什麼

| 腳本 | 內容 |
|---|---|
| `setup.bat` | 檢查 Python 3.10+（64 位元）、Node.js 18+、VC++ 執行階段 → 缺少的優先用 `winget` 安裝，沒有 winget 就從官網下載安裝檔 → 建立 `.venv` → `pip install` → 確認套件載入得了 → `npm install` 和 `npm run build` |
| `run_experiments.bat` | `prepare_data.py` → `build_features.py` → `evaluate.py` |
| `start.bat` | 啟動 FastAPI（`uvicorn`），自動打開瀏覽器 |
