"""Evaluate retrieval quality and the 'refuse when not in the papers' guard.

Usage:  python -m eval.evaluate            (retrieval only, no API key needed)
        python -m eval.evaluate --answers  (also generates answers and measures quote verification)
Writes eval/results.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag.answer import answer_question
from rag.index import get_index
from rag.retrieve import dense_search, embed_texts, is_answerable, load_models, retrieve

from .metrics import summarise

HERE = Path(__file__).resolve().parent


def run_evaluation(encoder, reranker, chunks, matrix, api_key: str | None = None, answers: bool = False) -> str:
    """Run the benchmark with already-loaded models and return the results as Markdown."""
    texts = [c.text for c in chunks]
    questions = [json.loads(l) for l in (HERE / "questions.jsonl").read_text().splitlines() if l.strip()]

    rows_dense, rows_rerank, verify = [], [], []
    for q in questions:
        qv = embed_texts(encoder, [q["q"]])[0]
        dense_hits = dense_search(qv, matrix, 10)
        hits = retrieve(q["q"], encoder, reranker, matrix, texts, candidates=25, final_k=10)
        base = {"gold": q["paper"], "answerable": q["answerable"]}
        rows_dense.append({**base, "ranked": [chunks[h.chunk_index].paper_id for h in dense_hits],
                           "pred_answerable": is_answerable([type(h)(h.chunk_index, h.dense_score) for h in dense_hits])})
        rows_rerank.append({**base, "ranked": [chunks[h.chunk_index].paper_id for h in hits],
                            "pred_answerable": is_answerable(hits)})
        if answers and q["answerable"] and is_answerable(hits):
            verify.append(answer_question(q["q"], [texts[h.chunk_index] for h in hits[:5]], api_key))

    dense, rer = summarise(rows_dense), summarise(rows_rerank)
    n_ans = sum(q["answerable"] for q in questions)
    lines = ["# Evaluation results", "",
             f"{dense['questions']} questions ({n_ans} answerable from the papers, "
             f"{dense['questions'] - n_ans} deliberately off-topic). Chunks in the index: {len(chunks)}.", "",
             "| Metric | Bi-encoder only | + Cross-encoder rerank |", "|---|---|---|"]
    for key in ["recall@1", "recall@3", "recall@5", "mrr", "refusal_accuracy"]:
        lines.append(f"| {key} | {dense[key]} | {rer[key]} |")
    if verify:
        claims = sum(len(a.claims) + len(a.removed) for a in verify)
        ok = sum(len(a.claims) for a in verify)
        answered = sum(1 for a in verify if a.answerable)
        errors = sum(1 for a in verify if a.error)
        lines += ["", f"Answers: {answered}/{len(verify)} answerable questions got a verified answer. "
                      f"Citation check: {ok}/{claims} final claims had quotes found in the cited source"
                      f" ({(ok / claims * 100 if claims else 0):.0f}%). Answer errors: {errors}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also call the LLM and check citations")
    args = ap.parse_args()
    encoder, reranker = load_models()
    chunks, matrix = get_index(encoder)
    md = run_evaluation(encoder, reranker, chunks, matrix, os.getenv("GROQ_API_KEY"), args.answers)
    (HERE / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
