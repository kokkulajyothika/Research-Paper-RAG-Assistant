# Evaluation results

Run on 30 October 2026 against the deployed app (Streamlit Community Cloud, CPU), using the real PyTorch models and the full index.

20 questions (17 answerable from the papers, 3 deliberately off-topic). Chunks in the index: 1289.

| Metric | Bi-encoder only | + Cross-encoder rerank |
|---|---|---|
| Recall@1 | 0.941 | **1.000** |
| Recall@3 | 0.941 | **1.000** |
| Recall@5 | 1.000 | 1.000 |
| MRR | 0.956 | **1.000** |
| Refusal accuracy | 1.000 | 1.000 |

- Recall@k: a passage from the correct paper is in the top k results.
- MRR: mean reciprocal rank of the first passage from the correct paper.
- Refusal accuracy: the guard answers the 17 answerable questions and refuses the 3 off-topic ones.

Answer-level citation checking (`--answers`) was not completed in this run: the free Groq tier rate-limits a full
batch, and the first attempt exposed a bug (Groq rejecting some schema-formatted outputs with HTTP 400) that has since
been fixed with a JSON-mode fallback. The question set is small (20), so these numbers show the pipeline works,
not that it is production-grade.
