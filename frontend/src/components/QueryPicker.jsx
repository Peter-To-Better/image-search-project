import { useEffect, useRef, useState } from "react";
import { getSamples, imgUrl } from "../api.js";

const SOURCES = [
  { id: "upload", label: "上傳圖片", hint: "自己的照片、手機拍的都可以" },
  { id: "semantic", label: "查詢集", hint: "不在圖庫裡的圖 → 測語意相似" },
  { id: "gallery", label: "圖庫原圖", hint: "搭配變形 → 測近似副本搜尋" },
];

export default function QueryPicker({ query, setQuery, transform, setTransform, transforms }) {
  const [source, setSource] = useState("semantic");
  const [samples, setSamples] = useState([]);
  const [seed, setSeed] = useState(1);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef();

  useEffect(() => {
    if (source === "upload") return;
    getSamples(source, 12, seed).then(setSamples).catch(() => setSamples([]));
  }, [source, seed]);

  // 支援 Ctrl/Cmd+V 貼上圖片
  useEffect(() => {
    const onPaste = (e) => {
      const f = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
      if (f) pickFile(f);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  });

  function pickFile(file) {
    if (!file) return;
    setSource("upload");
    setQuery({ type: "file", file, preview: URL.createObjectURL(file), label: file.name });
  }

  const isSel = (s) =>
    (query?.type === "gallery" && s.gallery_id === query.gallery_id) ||
    (query?.type === "query" && s.path === query.path);

  return (
    <div className="card picker">
      <h3>① 選擇查詢圖</h3>
      <div className="seg">
        {SOURCES.map((s) => (
          <button key={s.id} className={source === s.id ? "on" : ""} onClick={() => setSource(s.id)} title={s.hint}>
            {s.label}
          </button>
        ))}
      </div>
      <p className="hint">{SOURCES.find((s) => s.id === source).hint}</p>

      {source === "upload" ? (
        <div
          className={`dropzone ${drag ? "drag" : ""}`}
          onClick={() => fileRef.current.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            pickFile(e.dataTransfer.files[0]);
          }}
        >
          {query?.type === "file" ? (
            <img src={query.preview} alt="" />
          ) : (
            <span>
              拖放圖片到這裡、點一下選檔案
              <br />
              或直接 ⌘V / Ctrl+V 貼上
            </span>
          )}
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => pickFile(e.target.files[0])} />
        </div>
      ) : (
        <>
          <div className="sample-grid">
            {samples.map((s) => (
              <button
                key={s.path}
                className={`sample ${isSel(s) ? "sel" : ""}`}
                onClick={() =>
                  setQuery(
                    source === "gallery"
                      ? { type: "gallery", gallery_id: s.gallery_id, preview: imgUrl(s.path), label: s.label }
                      : { type: "query", path: s.path, preview: imgUrl(s.path), label: s.label }
                  )
                }
              >
                <img src={imgUrl(s.path)} alt={s.label} loading="lazy" />
              </button>
            ))}
          </div>
          <button className="link" onClick={() => setSeed((x) => x + 1)}>
            ↻ 換一批
          </button>
        </>
      )}

      <h3 style={{ marginTop: 18 }}>② 影像變形（測穩健性）</h3>
      <select value={transform} onChange={(e) => setTransform(e.target.value)}>
        {transforms.map((t) => (
          <option key={t.name} value={t.name}>
            {t.label}
          </option>
        ))}
      </select>
    </div>
  );
}
