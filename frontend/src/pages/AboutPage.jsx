const FEATURES = [
  ["aHash / dHash / pHash", "感知雜湊", "縮圖 → 灰階 → (DCT) → 與平均/中位數比較得到 64 bit", "極快、每張只要 8 bytes", "只能找近似副本；怕旋轉、裁切、翻轉"],
  ["HSV 顏色直方圖", "全域手工特徵", "統計 8×4×4 個顏色區間的像素比例，Hellinger 轉換", "簡單、不怕旋轉翻轉", "怕亮度改變；完全不懂形狀與語意"],
  ["SIFT + 視覺詞袋", "局部特徵", "SIFT 關鍵點 → k-means 建 1000 個視覺單字 → TF-IDF", "對旋轉、裁切、遮擋穩健", "慢；語意能力有限；圖太小時抓不到關鍵點"],
  ["ResNet-50", "深度學習 (CNN)", "ImageNet 預訓練，取 GAP 層 2048 維向量", "有語意理解，同類別可以找到", "向量大 (8 KB)、需要模型推論"],
  ["DINOv2 / CLIP", "自監督 ViT / 圖文多模態", "Transformer 的 CLS / 圖像嵌入；CLIP 可以文字搜圖", "目前最強的通用特徵之一", "模型大，需要 torch"],
];

const INDEXES = [
  ["暴力線性掃描", "O(N·d)", "O(N·d)", "精確", "baseline；N 小時 BLAS 向量化其實最快"],
  ["BK-tree", "平均 O(N log N)", "小半徑 ≈ O(log N)，大半徑 O(N)", "精確", "任何度量空間 (只需三角不等式)；適合 Hamming"],
  ["KD-tree", "O(N log N)", "低維 O(log N)，高維 O(N)", "精確", "展示維度災難"],
  ["倒排索引", "O(總詞數)", "O(Σ posting 長度)", "精確", "稀疏向量 (BoVW)；搜尋引擎同款"],
  ["LSH 隨機超平面", "O(N·L·K)", "次線性 O(N^ρ)", "近似", "有機率理論保證；參數 K、L 控制取捨"],
  ["IVF", "k-means O(N·nlist·d·iter)", "O((nlist + N·nprobe/nlist)·d)", "近似", "分群後只搜最近的幾群"],
  ["HNSW (FAISS)", "O(N log N)", "≈ O(log N)", "近似", "業界主流，工業級對照組"],
];

export default function AboutPage() {
  return (
    <div className="results">
      <div className="card">
        <h2>系統流程</h2>
        <div className="pipeline">
          <div className="step"><b>離線</b><span>圖庫 N 張圖</span></div>
          <div className="arrow">→</div>
          <div className="step"><b>特徵擷取</b><span>每張圖 → 向量 / 雜湊</span></div>
          <div className="arrow">→</div>
          <div className="step"><b>建立索引</b><span>BK-tree / LSH / IVF…</span></div>
        </div>
        <div className="pipeline">
          <div className="step hl"><b>線上</b><span>查詢圖</span></div>
          <div className="arrow">→</div>
          <div className="step hl"><b>同一個特徵擷取</b><span>q ∈ ℝ^d</span></div>
          <div className="arrow">→</div>
          <div className="step hl"><b>k-NN 搜尋</b><span>argmin dist(q, x)</span></div>
          <div className="arrow">→</div>
          <div className="step hl"><b>前 k 名結果</b><span>依距離排序</span></div>
        </div>
        <p className="note-text">
          核心問題：在度量空間 (X, dist) 中，給定查詢 q，找出距離最近的 k 個點 (k-Nearest Neighbors)。
          高維時精確解很難比 O(N) 更好 (維度災難)，所以實務上改用近似最近鄰 (ANN)，用一點點準確度換大量速度。
        </p>
      </div>

      <div className="card">
        <h2>特徵擷取方法</h2>
        <table>
          <thead><tr><th>方法</th><th>類別</th><th>做法</th><th>優點</th><th>缺點</th></tr></thead>
          <tbody>{FEATURES.map((r) => <tr key={r[0]}>{r.map((c, i) => <td key={i}>{i === 0 ? <b>{c}</b> : c}</td>)}</tr>)}</tbody>
        </table>
      </div>

      <div className="card">
        <h2>索引 / 搜尋結構</h2>
        <table>
          <thead><tr><th>結構</th><th>建立</th><th>查詢</th><th>精確?</th><th>備註</th></tr></thead>
          <tbody>{INDEXES.map((r) => <tr key={r[0]}>{r.map((c, i) => <td key={i}>{i === 0 ? <b>{c}</b> : c}</td>)}</tr>)}</tbody>
        </table>
      </div>

      <div className="card">
        <h2>關鍵理論</h2>
        <div className="two">
          <div className="theory">
            <h4>BK-tree 的剪枝 (三角不等式)</h4>
            <code>|d(q,x) − d(x,y)| ≤ d(q,y)</code>
            <p>要找 d(q,y) ≤ τ 的 y，只需進入邊標籤 e 滿足 d(q,x) − τ ≤ e ≤ d(q,x) + τ 的子樹。</p>
          </div>
          <div className="theory">
            <h4>隨機超平面 LSH 的碰撞機率</h4>
            <code>P[h(x) = h(y)] = 1 − θ(x,y)/π</code>
            <p>K 個串接 (AND) 後再用 L 張表 (OR)：P = 1 − (1 − p^K)^L，形成陡峭的 S 曲線，近的點幾乎必撞、遠的點幾乎不撞。</p>
          </div>
          <div className="theory">
            <h4>KD-tree 的維度災難</h4>
            <code>d → ∞：max dist / min dist → 1</code>
            <p>高維空間中所有點的距離都差不多，查詢點到切割超平面的距離幾乎總是小於第 k 近距離，無法剪枝。</p>
          </div>
          <div className="theory">
            <h4>評估指標</h4>
            <code>AP = Σ P@k · rel(k) / |Rel|</code>
            <p>mAP 同時考慮「找到多少」和「排得多前面」；ANN 另用 Recall@k (vs 精確解) 衡量近似品質。</p>
          </div>
        </div>
      </div>
    </div>
  );
}
