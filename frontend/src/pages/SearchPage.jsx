import { useEffect, useState } from "react";
import { search, textSearch } from "../api.js";
import QueryPicker from "../components/QueryPicker.jsx";
import ResultGrid, { precision } from "../components/ResultGrid.jsx";

export default function SearchPage({ info }) {
  const feats = info.features;
  const defaultFeat = feats.find((f) => f.name === "resnet50") || feats[0];
  const [query, setQuery] = useState(null);
  const [transform, setTransform] = useState("none");
  const [feature, setFeature] = useState(defaultFeat?.name);
  const [index, setIndex] = useState("brute");
  const [k, setK] = useState(12);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [text, setText] = useState("");

  const feat = feats.find((f) => f.name === feature);

  useEffect(() => {
    if (feat && !feat.indexes.includes(index)) setIndex("brute");
  }, [feature]); // eslint-disable-line

  // 任何設定改變就自動重新搜尋，Demo 時比較流暢
  useEffect(() => {
    if (!query || !feat || !feat.indexes.includes(index)) return;
    setBusy(true);
    setErr(null);
    search(query, { feature, index, k, transform })
      .then(setRes)
      .catch((e) => setErr(e.message))
      .finally(() => setBusy(false));
  }, [query, transform, feature, index, k]); // eslint-disable-line

  async function runText(e) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await textSearch(text, "brute", k);
      setRes({ ...r, query_image: null, truth: {}, text });
      setFeature("clip");
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  const families = [...new Set(feats.map((f) => f.family))];
  const p = res && precision(res.results, res.truth);

  return (
    <div className="layout">
      <aside>
        <QueryPicker {...{ query, setQuery, transform, setTransform }} transforms={info.transforms} />

        <div className="card">
          <h3>③ 特徵擷取方法</h3>
          {families.map((fam) => (
            <div key={fam} className="family">
              <div className="family-name">{fam}</div>
              {feats
                .filter((f) => f.family === fam)
                .map((f) => (
                  <label key={f.name} className={`opt ${feature === f.name ? "on" : ""}`}>
                    <input type="radio" checked={feature === f.name} onChange={() => setFeature(f.name)} />
                    {f.label}
                  </label>
                ))}
            </div>
          ))}

          <h3 style={{ marginTop: 18 }}>④ 索引 / 搜尋結構</h3>
          {feat?.indexes.map((i) => {
            const meta = info.indexes[i];
            return (
              <label key={i} className={`opt ${index === i ? "on" : ""}`}>
                <input type="radio" checked={index === i} onChange={() => setIndex(i)} />
                <span className="grow">{meta.label}</span>
                <span className={`tag ${meta.exact ? "exact" : "approx"}`}>{meta.exact ? "精確" : "近似"}</span>
                <span className="cx">{meta.complexity}</span>
              </label>
            );
          })}

          <h3 style={{ marginTop: 18 }}>回傳前 k 名：{k}</h3>
          <input type="range" min="4" max="40" step="4" value={k} onChange={(e) => setK(+e.target.value)} />
        </div>

        {info.text_search && (
          <form className="card" onSubmit={runText}>
            <h3>✦ 以文字搜圖 (CLIP)</h3>
            <div className="row">
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder="例：a photo of a red car" />
              <button className="primary">搜尋</button>
            </div>
          </form>
        )}
      </aside>

      <section className="main-col">
        {!res && !busy && (
          <div className="empty">
            <div className="big">◎</div>
            從左側選一張查詢圖開始
          </div>
        )}
        {err && <div className="error-box">{err}</div>}
        {res && (
          <>
            <div className="card query-summary">
              {res.query_image ? (
                <img src={res.query_image} className="qimg" alt="query" />
              ) : (
                <div className="qtext">“{res.text}”</div>
              )}
              <div className="stats">
                <Stat label="特徵擷取" value={`${res.encode_ms.toFixed(1)} ms`} />
                <Stat label="索引搜尋" value={`${res.search_ms.toFixed(2)} ms`} />
                <Stat
                  label="距離計算次數"
                  value={res.dist_computations != null ? res.dist_computations.toLocaleString() : "—"}
                  sub={
                    res.dist_computations != null
                      ? `圖庫的 ${((res.dist_computations / res.gallery_size) * 100).toFixed(1)}%`
                      : "FAISS 內部"
                  }
                />
                {p != null && (
                  <Stat label={`Precision@${res.results.length}`} value={`${(p * 100).toFixed(0)}%`} sub={`正解類別：${res.truth.label}`} />
                )}
              </div>
              {busy && <div className="spinner" />}
            </div>
            <div className="legend">
              <span className="sw exact" /> 原圖 <span className="sw same" /> 同類別 <span className="sw wrong" /> 不同類別
            </div>
            <ResultGrid results={res.results} truth={res.truth} kind={feat?.kind} />
          </>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}
