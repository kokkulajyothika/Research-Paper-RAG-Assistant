"""Evaluation metrics (pure Python so they can be unit-tested without models)."""
from __future__ import annotations


def recall_at_k(ranked_papers: list[str], gold: str, k: int) -> float:
    """1.0 if a passage from the correct paper is in the top k results, else 0.0."""
    return 1.0 if gold in ranked_papers[:k] else 0.0


def reciprocal_rank(ranked_papers: list[str], gold: str) -> float:
    """1 / rank of the first passage from the correct paper (0 if absent)."""
    for i, p in enumerate(ranked_papers, start=1):
        if p == gold:
            return 1.0 / i
    return 0.0


def refusal_accuracy(predicted_answerable: list[bool], expected_answerable: list[bool]) -> float:
    if not expected_answerable:
        return 0.0
    return sum(p == e for p, e in zip(predicted_answerable, expected_answerable)) / len(expected_answerable)


def summarise(rows: list[dict], k_values=(1, 3, 5)) -> dict:
    """rows: [{'ranked': [...paper ids...], 'gold': str|None, 'pred_answerable': bool, 'answerable': bool}]"""
    ans_rows = [r for r in rows if r["answerable"]]
    out = {f"recall@{k}": round(sum(recall_at_k(r["ranked"], r["gold"], k) for r in ans_rows) / len(ans_rows), 3)
           for k in k_values} if ans_rows else {}
    if ans_rows:
        out["mrr"] = round(sum(reciprocal_rank(r["ranked"], r["gold"]) for r in ans_rows) / len(ans_rows), 3)
    out["refusal_accuracy"] = round(refusal_accuracy([r["pred_answerable"] for r in rows], [r["answerable"] for r in rows]), 3)
    out["questions"] = len(rows)
    return out
