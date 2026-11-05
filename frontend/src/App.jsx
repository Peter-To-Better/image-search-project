import { useEffect, useState } from "react";
import { getInfo } from "./api.js";
import SearchPage from "./pages/SearchPage.jsx";
import ComparePage from "./pages/ComparePage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";
import AboutPage from "./pages/AboutPage.jsx";

const TABS = [
  { id: "search", label: "以圖搜圖" },
  { id: "compare", label: "方法比較" },
  { id: "results", label: "實驗結果" },
  { id: "about", label: "演算法說明" },
];

export default function App() {
  const [tab, setTab] = useState("search");
  const [info, setInfo] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    getInfo().then(setInfo).catch((e) => setErr(e.message));
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◎</span>
          <div>
            <h1>以圖搜圖</h1>
            <p>Content-Based Image Retrieval · 演算法專題</p>
          </div>
        </div>
        <nav>
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
        {info && <div className="gallery-size">圖庫 {info.gallery_size.toLocaleString()} 張 · {info.classes.length} 類</div>}
      </header>

      <main>
        {err && (
          <div className="error-box">
            無法連到後端：{err}
            <br />
            請確認已在 <code>backend/</code> 執行 <code>uvicorn server:app --port 8000</code>
          </div>
        )}
        {!info && !err && <div className="loading">載入中…</div>}
        {info && tab === "search" && <SearchPage info={info} />}
        {info && tab === "compare" && <ComparePage info={info} />}
        {tab === "results" && <ResultsPage />}
        {tab === "about" && <AboutPage />}
      </main>
    </div>
  );
}
