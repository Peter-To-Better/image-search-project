import { useEffect, useState } from "react";
import { compare } from "../api.js";
import QueryPicker from "../components/QueryPicker.jsx";
import ResultGrid, { judge, precision } from "../components/ResultGrid.jsx";

export default function ComparePage({ info }) {
  const [query, setQuery] = useState(null);
  const [transform, setTransform] = useState("none");
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!query) return;
    setBusy(true);
    setErr(null);
    compare(query, { k: 8, transform })
      .then(setRes)
      .catch((e) => setErr(e.message))
      .finally(() => setBusy(false));
  }, [query, transform]);

  const kindOf = Object.fromEntries(info.features.map((f) => [f.name, f.kind]));

  return (
    <div className="layout">
      <aside>
        <QueryPicker {...{ query, setQuery, transform, setTransform }} transforms={info.transforms} />
        <div className="card note">
          <b>玩法建議</b>
          <ul>
            <li>選「圖庫原圖」＋「旋轉」：感知雜湊找不到原圖，SIFT、CNN 找得到</li>
            <li>選「圖庫原圖」＋「調亮度」：顏色直方圖失效，雜湊仍然穩</li>
            <li>選「查詢集」：只有深度特徵能找到同類別的圖</li>
          </ul>
        </div>
      </aside>
      <section className="main-col">
        {!res && !busy && (
          <div className="empty">
            <div className="big">⇄</div>
            同一張查詢圖，所有特徵並排比較（皆使用精確的暴力搜尋）
          </div>
        )}
        {err && <div className="error-box">{err}</div>}
        {res && (
          <>
            <div className="card query-summary">
              <img src={res.query_image} className="qimg" alt="" />
              <div>
                <div className="stat-label">正確答案</div>
                <div className="stat-value small">
                  {res.truth.source_id != null ? `原圖 #${res.truth.source_id}・` : ""}
                  {res.truth.label || "（上傳圖片，無標準答案）"}
                </div>
              </div>
              {busy && <div className="spinner" />}
            </div>
            {res.runs.map((r) => {
              const p = precision(r.results, res.truth);
              const foundExact = r.results.findIndex((x) => judge(x, res.truth) === "exact");
              return (
                <div className="card compare-row" key={r.feature}>
                  <div className="compare-head">
                    <b>{r.label}</b>
                    <span className="muted">
                      {r.encode_ms.toFixed(0)} ms 擷取 · {r.search_ms.toFixed(2)} ms 搜尋
                    </span>
                    {p != null && <span className="pill">P@{r.results.length} = {(p * 100).toFixed(0)}%</span>}
                    {res.truth.source_id != null && (
                      <span className={`pill ${foundExact >= 0 ? "ok" : "bad"}`}>
                        {foundExact >= 0 ? `原圖排第 ${foundExact + 1}` : "沒找到原圖"}
                      </span>
                    )}
                  </div>
                  <ResultGrid results={r.results} truth={res.truth} compact kind={kindOf[r.feature]} />
                </div>
              );
            })}
          </>
        )}
      </section>
    </div>
  );
}
