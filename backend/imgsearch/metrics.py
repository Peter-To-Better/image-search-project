"""檢索評估指標。

ranked   : 系統回傳的排序結果 (圖庫 id 的 list)
relevant : 正確答案的集合

Precision@k = 前 k 名中正確的比例
Recall@k    = 正確答案中有多少出現在前 k 名
AP          = Σ_k P@k · rel(k) / |relevant|   (每找到一個正確答案時的 precision 平均)
mAP         = 所有查詢 AP 的平均
"""
import numpy as np


def precision_at_k(ranked, relevant, k):
    return sum(1 for r in ranked[:k] if r in relevant) / k


def recall_at_k(ranked, relevant, k):
    return sum(1 for r in ranked[:k] if r in relevant) / max(1, len(relevant))


def hit_at_k(ranked, relevant, k):
    return float(any(r in relevant for r in ranked[:k]))


def average_precision(ranked, relevant):
    hits, s = 0, 0.0
    for i, r in enumerate(ranked, 1):
        if r in relevant:
            hits += 1
            s += hits / i
    return s / max(1, len(relevant))


def ann_recall(approx, exact):
    """近似搜尋品質 (Recall@k vs 精確解)。

    approx / exact 為 [(id, dist), ...]。
    Hamming 距離是整數，常有很多張圖與第 k 名「同分」，只比對 id 會誤判，
    所以改成：近似結果中距離 ≤ 精確解第 k 名距離 的比例 (tie-aware)。
    """
    if not exact:
        return 1.0
    kth = exact[-1][1] + 1e-6
    return sum(1 for _, d in approx[:len(exact)] if d <= kth) / len(exact)


def summarize(values):
    return float(np.mean(values)) if len(values) else 0.0
