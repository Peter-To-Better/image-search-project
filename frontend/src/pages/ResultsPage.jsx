import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { getResults } from "../api.js";

const C = ["#4f7cff", "#16b89a", "#f5a524", "#e5484d", "#9b5de5", "#00a6d6", "#8a8f98", "#ff7ab6"];
const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const TF = {
  rotate: "旋轉", crop: "裁切", brightness: "亮度", jpeg: "JPEG", blur_scale: "縮小模糊",
  watermark: "浮水印", flip: "翻轉",
};

export default function ResultsPage() {
  const [R, setR] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    getResults().then(setR).catch((e) => setErr(e.message));
  }, []);
  if (err) return <div className="error-box">{err}（請先執行 <code>python scripts/evaluate.py</code>）</div>;
  if (!R) return <div className="loading">載入中…</div>;

  const acc = R.accuracy || {};
  const accRows = Object.entries(acc).map(([k, v]) => ({ name: k, ...v }));

  return (
    <div className="results">
      <div className="card">
        <h2>實驗設定</h2>
        <p>
          資料集 Caltech-101：{R.meta.classes} 類、圖庫 <b>{R.meta.gallery}</b> 張；語意查詢{" "}
          <b>{R.meta.semantic_queries}</b> 張（不在圖庫中，正解為同類別）；近似副本查詢{" "}
          <b>{R.meta.duplicate_queries}</b> 張（圖庫原圖經 {R.meta.transforms.length} 種變形，正解為原圖）。
          <span className="muted"> 執行時間 {R.meta.time}</span>
        </p>
      </div>

      {accRows.length > 0 && (
        <Section title="實驗 1｜各特徵的檢索準確度" note="語意 mAP 衡量「找到同類別」的能力；近似副本 Recall@1 衡量「找到原圖」的能力。">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={accRows}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} />
              <Tooltip formatter={(v) => pct(v)} />
              <Legend />
              <Bar isAnimationActive={false} dataKey="sem_mAP" name="語意 mAP" fill={C[0]} radius={[4, 4, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="sem_P@10" name="語意 P@10" fill={C[1]} radius={[4, 4, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="dup_R@1" name="近似副本 Recall@1" fill={C[2]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <table>
            <thead>
              <tr>
                <th>特徵</th><th>類別</th><th>維度</th><th>每張大小</th><th>擷取時間/張</th>
                <th>語意 mAP</th><th>語意 P@10</th><th>副本 R@1</th><th>副本 R@10</th>
              </tr>
            </thead>
            <tbody>
              {accRows.map((r) => (
                <tr key={r.name}>
                  <td><b>{r.label}</b></td><td>{r.family}</td><td>{r.dim}{r.dim === 64 && " bit"}</td>
                  <td>{r.bytes_per_image < 1024 ? `${r.bytes_per_image} B` : `${(r.bytes_per_image / 1024).toFixed(1)} KB`}</td>
                  <td>{r.extract_ms_per_image.toFixed(1)} ms</td>
                  <td>{pct(r.sem_mAP)}</td><td>{pct(r["sem_P@10"])}</td><td>{pct(r["dup_R@1"])}</td><td>{pct(r["dup_R@10"])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}

      {R.robustness && (
        <Section title="實驗 2｜穩健性：每種變形下找回原圖的比例 (Recall@1)" note="感知雜湊對旋轉、裁切、翻轉幾乎完全失效；顏色直方圖怕亮度改變；SIFT 和 CNN 對幾何變形穩健，但影像縮得太小時會失效。">
          <table className="heat">
            <thead>
              <tr>
                <th>特徵</th>
                {R.meta.transforms.map((t) => <th key={t}>{TF[t] || t}</th>)}
              </tr>
            </thead>
            <tbody>
              {Object.entries(R.robustness).map(([f, row]) => (
                <tr key={f}>
                  <td><b>{f}</b></td>
                  {R.meta.transforms.map((t) => (
                    <td key={t} style={{ background: heat(row[t]), color: row[t] > 0.55 ? "#fff" : "inherit" }}>
                      {(row[t] * 100).toFixed(0)}%
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}

      {R.indexes && (
        <Section title="實驗 3｜特徵 × 索引結構" note="「距離計算比例」與機器無關，最能反映演算法本身的剪枝效果；查詢時間則受常數與向量化影響 (numpy 的暴力法用 BLAS，常數極小)。">
          <table>
            <thead>
              <tr><th>特徵</th><th>索引</th><th>建索引</th><th>查詢時間</th><th>距離計算比例</th><th>Recall@10 (vs 精確解)</th></tr>
            </thead>
            <tbody>
              {R.indexes.map((r, i) => (
                <tr key={i} className={r.index === "brute" ? "sep" : ""}>
                  <td>{r.index === "brute" ? <b>{r.feature}</b> : ""}</td>
                  <td>{r.label}</td>
                  <td>{r.build_s < 1 ? `${(r.build_s * 1000).toFixed(0)} ms` : `${r.build_s.toFixed(2)} s`}</td>
                  <td>{r.query_ms.toFixed(2)} ms</td>
                  <td><Meter v={r.dist_frac} /></td>
                  <td><Meter v={r["ann_recall@10"]} good /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}

      {R.scaling && (
        <Section title="實驗 4｜擴展性：圖庫大小 N 與查詢時間 (log-log)" note="斜率代表複雜度的次方：暴力法斜率 ≈ 1 (線性)，HNSW 幾乎是水平線 (≈ log N)。N > 7,942 的部分以真實向量加雜訊合成。">
          <div className="two">
            {R.scaling.dense && (
              <ScalingChart rows={R.scaling.dense.rows} keys={["brute", "lsh", "ivf", "hnsw"]} title={`${R.scaling.dense.feature}（2048 維向量）`} />
            )}
            {R.scaling.hash && (
              <ScalingChart rows={R.scaling.hash.rows} keys={["brute", "bktree"]} title="pHash（64-bit）" />
            )}
          </div>
          {R.scaling.dense && (
            <table>
              <thead>
                <tr><th>N</th>{["brute", "lsh", "ivf", "hnsw"].map((k) => <th key={k}>{k} ms (recall)</th>)}</tr>
              </thead>
              <tbody>
                {R.scaling.dense.rows.map((r) => (
                  <tr key={r.N}>
                    <td>{r.N.toLocaleString()}</td>
                    {["brute", "lsh", "ivf", "hnsw"].map((k) => (
                      <td key={k}>{r[k] != null ? `${r[k].toFixed(2)} (${pct(r[k + "_recall"])})` : "—"}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      )}

      {R.lsh && <LshSection L={R.lsh} />}

      {R.kdtree && (
        <Section title="實驗 6｜KD-tree 的維度災難" note={`把 ${R.kdtree.feature} 特徵用 PCA 降到不同維度後建 KD-tree。低維時只需看極少數點；維度升高後幾乎每個點都要算，剪枝失效。`}>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={R.kdtree.rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="dim" scale="log" type="number" domain={[2, 256]} ticks={[2, 4, 8, 16, 32, 64, 128, 256]} label={{ value: "維度", position: "insideBottomRight", offset: -4 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} />
              <Tooltip formatter={(v) => pct(v)} labelFormatter={(d) => `${d} 維`} />
              <Line isAnimationActive={false} dataKey="visited_frac" name="需計算距離的點比例" stroke={C[3]} strokeWidth={2.5} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </Section>
      )}

      {R.bktree && (
        <Section title="實驗 7｜BK-tree 搜尋半徑與剪枝" note={`pHash，N=${R.bktree.N}，樹深 ${R.bktree.depth}。半徑小 (找幾乎相同的圖) 時只走訪極少數節點；半徑一大，三角不等式的剪枝範圍變寬，退化成線性掃描。`}>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={R.bktree.rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="radius" type="number" label={{ value: "Hamming 半徑", position: "insideBottomRight", offset: -4 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} />
              <Tooltip formatter={(v, n) => (n === "平均結果數" ? v.toFixed(1) : pct(v))} labelFormatter={(r) => `半徑 ${r}`} />
              <Line isAnimationActive={false} dataKey="visited_frac" name="走訪節點比例" stroke={C[0]} strokeWidth={2.5} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </Section>
      )}
    </div>
  );
}

function Section({ title, note, children }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      {note && <p className="note-text">{note}</p>}
      {children}
    </div>
  );
}

function Meter({ v, good }) {
  if (v == null) return <span className="muted">—</span>;
  return (
    <span className="meter">
      <span className="bar" style={{ width: `${Math.min(100, v * 100)}%`, background: good ? C[1] : C[0] }} />
      <span className="val">{pct(v)}</span>
    </span>
  );
}

function heat(v) {
  const a = Math.max(0, Math.min(1, v));
  return `rgba(79,124,255,${0.08 + a * 0.85})`;
}

function ScalingChart({ rows, keys, title }) {
  return (
    <div>
      <div className="chart-title">{title}</div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="N" type="number" scale="log" domain={["dataMin", "dataMax"]} ticks={rows.map((r) => r.N)} tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} />
          <YAxis scale="log" domain={["auto", "auto"]} tickFormatter={(v) => (v < 1 ? v.toFixed(2) : v.toFixed(0))} label={{ value: "ms/查詢", angle: -90, position: "insideLeft" }} />
          <Tooltip formatter={(v) => `${v.toFixed(3)} ms`} labelFormatter={(n) => `N = ${n.toLocaleString()}`} />
          <Legend />
          {keys.filter((k) => rows[0][k] != null).map((k, i) => (
            <Line isAnimationActive={false} key={k} dataKey={k} stroke={C[i]} strokeWidth={2.5} dot={{ r: 3 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function LshSection({ L }) {
  const groups = [];
  for (const mp of [0, 1])
    for (const t of [1, 4, 8, 16]) {
      const rows = L.rows.filter((r) => r.L === t && r.multiprobe === mp);
      if (rows.length) groups.push({ name: `L=${t}${mp ? " +multi-probe" : ""}`, rows, mp });
    }
  const theory = L.theory.sims.map((s, i) => {
    const o = { sim: s };
    Object.entries(L.theory.curves).forEach(([k, v]) => (o[k] = v[i]));
    return o;
  });
  return (
    <Section title="實驗 5｜LSH 參數取捨：K (每張表的 bit 數) × L (表的數量)" note={`每個點是一組 (K, L)。X 軸為需要精確計算的候選比例 (越小越快)，Y 軸為與精確解相比的 Recall@10。K 越大桶越細、候選越少但容易漏；L 越多越不容易漏但越慢；multi-probe 可以用較少的表達到相同召回率。暴力法查詢約 ${L.brute_ms.toFixed(2)} ms。`}>
      <div className="two">
        <div>
          <div className="chart-title">候選比例 vs 召回率（實測）</div>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="cand_frac" type="number" scale="log" domain={[0.0005, 1]} tickFormatter={(v) => `${(v * 100).toPrecision(2)}%`} name="候選比例" />
              <YAxis dataKey="recall@10" type="number" domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} name="Recall@10" />
              <ZAxis dataKey="K" range={[30, 30]} name="K" />
              <Tooltip formatter={(v, n) => (n === "K" ? v : pct(v))} />
              <Legend />
              {groups.map((g, i) => (
                <Scatter isAnimationActive={false} key={g.name} name={g.name} data={g.rows} fill={C[i % 4]} line shape={g.mp ? "triangle" : "circle"} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="chart-title">理論碰撞機率 P = 1 − (1 − (1 − θ/π)^K)^L</div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={theory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="sim" type="number" domain={[0, 1]} label={{ value: "cos 相似度", position: "insideBottomRight", offset: -4 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(v) => v.toFixed(3)} labelFormatter={(s) => `cos = ${s.toFixed(2)}`} />
              <Legend />
              {Object.keys(L.theory.curves).map((k, i) => (
                <Line isAnimationActive={false} key={k} dataKey={k} stroke={C[i]} dot={false} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Section>
  );
}
