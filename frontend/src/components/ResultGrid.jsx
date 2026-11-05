import { imgUrl } from "../api.js";

// 判斷每個結果是否「正確」：與原圖同一張 (近似副本) 或同類別 (語意)
export function judge(r, truth) {
  if (truth?.source_id != null && r.id === truth.source_id) return "exact";
  if (truth?.label && r.label === truth.label) return "same";
  return truth?.label ? "wrong" : "unknown";
}

export function precision(results, truth) {
  if (!truth?.label || !results.length) return null;
  return results.filter((r) => judge(r, truth) !== "wrong").length / results.length;
}

export default function ResultGrid({ results, truth, compact = false, kind }) {
  return (
    <div className={`result-grid ${compact ? "compact" : ""}`}>
      {results.map((r, i) => {
        const j = judge(r, truth);
        return (
          <figure key={r.id} className={`result ${j}`} title={`${r.label}  #${r.id}`}>
            <div className="thumb">
              <img src={imgUrl(r.path)} alt={r.label} loading="lazy" />
              <span className="rank">{i + 1}</span>
              {j === "exact" && <span className="badge">原圖</span>}
            </div>
            {!compact && (
              <figcaption>
                <span className="lbl">{r.label}</span>
                <span className="dist">
                  {kind === "hash" ? `Hamming ${r.distance}` : `sim ${(1 - r.distance).toFixed(3)}`}
                </span>
              </figcaption>
            )}
          </figure>
        );
      })}
    </div>
  );
}
