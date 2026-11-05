"""步驟 3：跑完所有實驗，結果寫到 results/results.json，圖表存成 results/*.png。

實驗：
  1. accuracy   各特徵的檢索準確度 (語意查詢 mAP / P@10；近似副本 Recall@1 / @10)
  2. robustness 各特徵在每一種影像變形下的 Recall@1
  3. indexes    各「特徵 × 索引」組合的查詢時間、距離計算次數、ANN Recall@10
  4. scaling    圖庫大小 N 與查詢時間的關係 (O(N) vs 次線性)
  5. lsh        LSH 參數 (K bits, L tables, multi-probe) 的召回率 / 速度取捨
  6. kdtree     KD-tree 在不同維度下的剪枝效率 → 維度災難
  7. bktree     BK-tree 搜尋半徑與走訪節點比例

用法：python scripts/evaluate.py            # 全部
      python scripts/evaluate.py --only lsh kdtree
"""
import argparse
import json
import time

import numpy as np
from tqdm import tqdm

import _path  # noqa: F401
from imgsearch import features as F
from imgsearch.config import FEATURE_DIR, RESULTS_DIR, ROOT
from imgsearch.dataset import load_split, open_image
from imgsearch.engine import INDEXES, index_options, make_index
from imgsearch.hashing import hamming_many
from imgsearch.indexes import (HNSW, IVF, BKTree, BruteForceCosine, BruteForceHamming, KDTree,
                               RandomHyperplaneLSH, faiss_available)
from imgsearch.metrics import ann_recall, average_precision, hit_at_k, precision_at_k
from imgsearch.transforms import TRANSFORMS

RES_FILE = RESULTS_DIR / "results.json"
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- 工具
def query_features(name, split, X_gallery):
    """計算 (並快取) 所有查詢圖的特徵。"""
    f = FEATURE_DIR / f"queries_{name}.npz"
    if f.exists():
        d = np.load(f)
        return d["sem"], d["dup"]
    enc = F.QueryEncoder(name, F.load(name))
    out = {}
    for key, qs in (("sem", split["semantic_queries"]), ("dup", split["duplicate_queries"])):
        feats = []
        for i in tqdm(range(0, len(qs), 32), desc=f"query {name}/{key}"):
            feats.append(np.asarray(enc.batch([open_image(q["path"]) for q in qs[i:i + 32]]),
                                    dtype=X_gallery.dtype))
        out[key] = np.concatenate(feats)
    np.savez(f, **out)
    return out["sem"], out["dup"]


def full_ranking(kind, X, q):
    if kind == "hash":
        d = hamming_many(int(q), X)
        return np.lexsort((np.arange(len(d)), d))
    return np.argsort(-(X @ q), kind="stable")


def timed(fn, qs, k=10):
    out, t0 = [], time.perf_counter()
    for q in qs:
        out.append(fn(q, k))
    return out, (time.perf_counter() - t0) / len(qs) * 1000


# ---------------------------------------------------------------- 1 + 2
def exp_accuracy(split, feats):
    labels = np.array([g["label"] for g in split["gallery"]])
    acc, rob = {}, {}
    for name in feats:
        X = F.load(name)["X"]
        kind = F.FEATURES[name]["kind"]
        sem, dup = query_features(name, split, X)
        aps, p10 = [], []
        for q, meta in zip(sem, split["semantic_queries"]):
            rank = full_ranking(kind, X, q)
            rel = set(np.where(labels == meta["label"])[0].tolist())
            aps.append(average_precision(rank.tolist(), rel))
            p10.append(precision_at_k(rank[:10].tolist(), rel, 10))
        r1, r10, rr = [], [], []
        per_t = {t: [] for t in TRANSFORMS}
        for q, meta in zip(dup, split["duplicate_queries"]):
            rank = full_ranking(kind, X, q)[:100].tolist()
            rel = {meta["source_id"]}
            h1 = hit_at_k(rank, rel, 1)
            r1.append(h1)
            r10.append(hit_at_k(rank, rel, 10))
            pos = rank.index(meta["source_id"]) + 1 if meta["source_id"] in rank else None
            rr.append(1 / pos if pos else 0)
            per_t[meta["transform"]].append(h1)
        acc[name] = {"label": F.FEATURES[name]["label"], "family": F.FEATURES[name]["family"],
                     "sem_mAP": float(np.mean(aps)), "sem_P@10": float(np.mean(p10)),
                     "dup_R@1": float(np.mean(r1)), "dup_R@10": float(np.mean(r10)),
                     "dup_MRR": float(np.mean(rr)),
                     "dim": int(X.shape[1]) if X.ndim == 2 else 64,
                     "bytes_per_image": int(8 if kind == "hash" else X.shape[1] * 4),
                     "extract_ms_per_image": float(F.load(name)["seconds"]) / len(X) * 1000}
        rob[name] = {t: float(np.mean(v)) for t, v in per_t.items()}
        print(f"{name:9s} sem mAP={acc[name]['sem_mAP']:.3f} P@10={acc[name]['sem_P@10']:.3f} "
              f"dup R@1={acc[name]['dup_R@1']:.3f}")
    return acc, rob


# ---------------------------------------------------------------- 3
def exp_indexes(split, feats):
    out = []
    for name in feats:
        X = F.load(name)["X"]
        sem, dup = query_features(name, split, X)
        qs = np.concatenate([sem, dup[::3]])          # 300 語意 + 350 副本 查詢
        exact = None
        for idx in index_options(name):
            t0 = time.perf_counter()
            index = make_index(name, idx, X)
            build = time.perf_counter() - t0
            res, ms = timed(index.search, qs)
            ids = [r[0] for r in res]
            if idx == "brute":
                exact = ids
            dc = [r[1].get("dist_computations") for r in res]
            out.append({"feature": name, "index": idx, "label": INDEXES[idx]["label"],
                        "build_s": build, "query_ms": ms,
                        "dist_frac": (float(np.mean(dc)) / len(X)) if dc[0] is not None else None,
                        "ann_recall@10": float(np.mean([ann_recall(a, e) for a, e in zip(ids, exact)]))})
            print(f"  {name:9s} {idx:9s} build={build:6.2f}s query={ms:7.2f}ms "
                  f"recall={out[-1]['ann_recall@10']:.3f}")
    return out


# ---------------------------------------------------------------- 4
def exp_scaling(split, feats):
    out = {}
    if "resnet50" in feats or any(F.FEATURES[f]["kind"] == "dense" for f in feats):
        name = "resnet50" if "resnet50" in feats else [f for f in feats if F.FEATURES[f]["kind"] == "dense"][0]
        X0 = F.load(name)["X"]
        sem, _ = query_features(name, split, X0)
        qs = sem[:100]
        rows = []
        for N in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            X = _expand_dense(X0, N)
            row = {"N": N}
            methods = {"brute": lambda: BruteForceCosine(), "lsh": lambda: RandomHyperplaneLSH(10, 16, 1),
                       "ivf": lambda: IVF(nprobe=8)}
            if faiss_available():
                methods["hnsw"] = lambda: HNSW(M=16, ef_construction=80)
            exact = None
            for m, ctor in methods.items():
                idx = ctor().build(X)
                res, ms = timed(idx.search, qs)
                ids = [r[0] for r in res]
                exact = ids if m == "brute" else exact
                row[m] = ms
                row[m + "_recall"] = float(np.mean([ann_recall(a, e) for a, e in zip(ids, exact)]))
            rows.append(row)
            print("  scaling dense", row)
        out["dense"] = {"feature": name, "rows": rows}
    if "phash" in feats:
        H0 = F.load("phash")["X"]
        _, dup = query_features("phash", split, H0)
        qs = [int(x) for x in dup[:200]]
        rows = []
        for N in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            H = _expand_hash(H0, N)
            bf, bk = BruteForceHamming().build(H), BKTree().build(H.tolist())
            _, ms_bf = timed(bf.search, qs)
            res, ms_bk = timed(bk.search, qs)
            frac = float(np.mean([r[1]["dist_computations"] for r in res])) / N
            rows.append({"N": N, "brute": ms_bf, "bktree": ms_bk, "bktree_frac": frac})
            print("  scaling hash", rows[-1])
        out["hash"] = {"feature": "phash", "rows": rows}
    return out


def _expand_dense(X, N):
    """N 大於真實圖庫時，以「真實向量 + 雜訊」合成更多資料點 (模擬更大的圖庫)。"""
    if N <= len(X):
        return X[:N]
    extra = X[rng.integers(0, len(X), N - len(X))]
    extra = extra + rng.normal(0, 0.03, extra.shape).astype(np.float32)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    return np.concatenate([X, extra]).astype(np.float32)


def _expand_hash(H, N):
    if N <= len(H):
        return H[:N]
    base = H[rng.integers(0, len(H), N - len(H))]
    flips = np.zeros(len(base), np.uint64)
    for _ in range(8):                                # 每個合成雜湊隨機翻轉約 8 個 bit
        flips |= (np.uint64(1) << rng.integers(0, 64, len(base)).astype(np.uint64))
    return np.concatenate([H, base ^ flips])


# ---------------------------------------------------------------- 5
def exp_lsh(split, feats):
    name = "resnet50" if "resnet50" in feats else [f for f in feats if F.FEATURES[f]["kind"] == "dense"][0]
    X = F.load(name)["X"]
    sem, _ = query_features(name, split, X)
    qs = sem
    bf = BruteForceCosine().build(X)
    exact_res, bf_ms = timed(bf.search, qs)
    exact = [r[0] for r in exact_res]
    rows = []
    for K in [6, 8, 10, 12, 14, 16, 20]:
        for L in [1, 4, 8, 16]:
            for mp in [0, 1]:
                idx = RandomHyperplaneLSH(K, L, mp).build(X)
                res, ms = timed(idx.search, qs)
                ids = [r[0] for r in res]
                rows.append({"K": K, "L": L, "multiprobe": mp, "query_ms": ms,
                             "cand_frac": float(np.mean([r[1]["dist_computations"] for r in res])) / len(X),
                             "recall@10": float(np.mean([ann_recall(a, e) for a, e in zip(ids, exact)]))})
        print(f"  lsh K={K} done")
    # 理論碰撞機率曲線
    sims = np.linspace(0, 1, 51)
    theory = {f"K{K}_L{L}": RandomHyperplaneLSH.collision_prob(sims, K, L).tolist()
              for K, L in [(12, 1), (12, 4), (12, 10), (16, 10), (8, 10)]}
    # 真實資料中「第 10 近鄰」的餘弦相似度分布，說明為什麼某些參數有效
    s10 = [float(1 - r[0][-1][1]) for r in exact_res]
    return {"feature": name, "brute_ms": bf_ms, "rows": rows,
            "theory": {"sims": sims.tolist(), "curves": theory}, "kth_sim_mean": float(np.mean(s10))}


# ---------------------------------------------------------------- 6
def exp_kdtree(split, feats):
    name = "resnet50" if "resnet50" in feats else [f for f in feats if F.FEATURES[f]["kind"] == "dense"][0]
    X = F.load(name)["X"]
    sem, _ = query_features(name, split, X)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)       # PCA
    rows = []
    for d in [2, 4, 8, 16, 32, 64, 128, 256]:
        P = Vt[:d].T
        Xd = (X - mu) @ P
        Xd /= np.linalg.norm(Xd, axis=1, keepdims=True) + 1e-8
        qd = (sem[:150] - mu) @ P
        qd /= np.linalg.norm(qd, axis=1, keepdims=True) + 1e-8
        kd = KDTree(leaf_size=16).build(Xd.astype(np.float32))
        bf = BruteForceCosine().build(Xd)
        res, ms_kd = timed(kd.search, qd.astype(np.float32))
        _, ms_bf = timed(bf.search, qd.astype(np.float32))
        frac = float(np.mean([r[1]["dist_computations"] for r in res])) / len(X)
        rows.append({"dim": d, "visited_frac": frac, "kd_ms": ms_kd, "brute_ms": ms_bf})
        print(f"  kdtree dim={d:4d} visited={frac:.3f}")
    return {"feature": name, "N": len(X), "rows": rows}


# ---------------------------------------------------------------- 7
def exp_bktree(split, feats):
    H = F.load("phash")["X"]
    _, dup = query_features("phash", split, H)
    bk = BKTree().build(H.tolist())
    rows = []
    for r in [0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]:
        fr, found = [], []
        for q in dup[:300]:
            res, st = bk.range_search(int(q), r)
            fr.append(st["dist_computations"] / len(H))
            found.append(len(res))
        rows.append({"radius": r, "visited_frac": float(np.mean(fr)), "avg_results": float(np.mean(found))})
    return {"N": len(H), "depth": bk.depth(), "rows": rows}


# ---------------------------------------------------------------- 圖表
def plot(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})

    if "accuracy" in R:
        a = R["accuracy"]
        names = list(a)
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(9, 4))
        for j, (m, lab) in enumerate([("sem_mAP", "Semantic mAP"), ("sem_P@10", "Semantic P@10"),
                                      ("dup_R@1", "Near-dup Recall@1")]):
            ax.bar(x + (j - 1) * .27, [a[n][m] for n in names], .27, label=lab)
        ax.set_xticks(x, names)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("Retrieval accuracy by feature")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "accuracy.png")
        plt.close(fig)
    if "robustness" in R:
        rb = R["robustness"]
        names, ts = list(rb), list(TRANSFORMS)
        M = np.array([[rb[n][t] for t in ts] for n in names])
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(ts)), ts, rotation=30)
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(ts)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < .6 else "k", fontsize=8)
        fig.colorbar(im)
        ax.set_title("Near-duplicate Recall@1 under each transform")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "robustness.png")
        plt.close(fig)
    if "scaling" in R:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key in zip(axes, ["dense", "hash"]):
            if key not in R["scaling"]:
                continue
            rows = R["scaling"][key]["rows"]
            Ns = [r["N"] for r in rows]
            for m in [k for k in rows[0] if k not in ("N",) and not k.endswith(("_recall", "_frac"))]:
                ax.plot(Ns, [r[m] for r in rows], "o-", label=m)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("gallery size N")
            ax.set_ylabel("ms / query")
            ax.set_title(f"Scaling ({R['scaling'][key]['feature']})")
            ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "scaling.png")
        plt.close(fig)
    if "lsh" in R:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for mp, mk in [(0, "o"), (1, "^")]:
            for L in [1, 4, 8, 16]:
                rs = [r for r in R["lsh"]["rows"] if r["L"] == L and r["multiprobe"] == mp]
                ax.plot([r["query_ms"] for r in rs], [r["recall@10"] for r in rs], mk + "-",
                        label=f"L={L}{' +probe' if mp else ''}", alpha=.8)
        ax.axvline(R["lsh"]["brute_ms"], color="k", ls="--", label="brute force")
        ax.set_xscale("log")
        ax.set_xlabel("ms / query")
        ax.set_ylabel("Recall@10 vs exact")
        ax.set_title("LSH speed / recall trade-off (points: K bits)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "lsh_tradeoff.png")
        plt.close(fig)
    if "kdtree" in R:
        rows = R["kdtree"]["rows"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["dim"] for r in rows], [r["visited_frac"] for r in rows], "o-")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("dimension (PCA)")
        ax.set_ylabel("fraction of points visited")
        ax.set_title("KD-tree: curse of dimensionality")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "kdtree_dim.png")
        plt.close(fig)
    if "bktree" in R:
        rows = R["bktree"]["rows"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["radius"] for r in rows], [r["visited_frac"] for r in rows], "o-")
        ax.set_xlabel("search radius (Hamming)")
        ax.set_ylabel("fraction of nodes visited")
        ax.set_title("BK-tree pruning vs radius (pHash)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "bktree_radius.png")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+",
                    choices=["accuracy", "indexes", "scaling", "lsh", "kdtree", "bktree"])
    a = ap.parse_args()
    split = load_split()
    feats = F.available()
    print("可用特徵：", feats)
    R = json.loads(RES_FILE.read_text()) if RES_FILE.exists() else {}
    todo = a.only or ["accuracy", "indexes", "scaling", "lsh", "kdtree", "bktree"]
    R["meta"] = {"gallery": len(split["gallery"]), "semantic_queries": len(split["semantic_queries"]),
                 "duplicate_queries": len(split["duplicate_queries"]), "classes": len(split["classes"]),
                 "transforms": list(TRANSFORMS), "features": feats,
                 "time": time.strftime("%Y-%m-%d %H:%M")}
    for name in todo:
        print(f"\n######## {name} ########")
        if name == "accuracy":
            R["accuracy"], R["robustness"] = exp_accuracy(split, feats)
        elif name == "bktree" and "phash" not in feats:
            continue
        else:
            R[name] = globals()[f"exp_{name}"](split, feats)
        RES_FILE.write_text(json.dumps(R, ensure_ascii=False, indent=1))
    plot(R)
    print(f"\n結果已寫入 {RES_FILE.relative_to(ROOT)} 與 results/*.png")
