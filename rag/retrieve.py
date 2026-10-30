"""Two-stage retrieval: a PyTorch bi-encoder finds candidates, a PyTorch cross-encoder reranks them.

The models are injected as objects with `.encode()` / `.predict()` so the logic can be unit-tested
with small fake models, and the real sentence-transformers models are only loaded by `load_models()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class Hit:
    chunk_index: int
    dense_score: float
    rerank_score: float | None = None


def load_models():
    """Load the two PyTorch models (downloaded from Hugging Face on first run)."""
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.set_num_threads(2)
    return SentenceTransformer(EMBED_MODEL, device="cpu"), CrossEncoder(RERANK_MODEL, device="cpu")


def embed_texts(encoder, texts: list[str], batch_size: int = 64) -> np.ndarray:
    vecs = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def dense_search(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[Hit]:
    """Cosine similarity search (vectors are L2-normalised, so a dot product is the cosine)."""
    if matrix.size == 0:
        return []
    scores = matrix @ query_vec.reshape(-1)
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [Hit(int(i), float(scores[i])) for i in top]


def rerank(reranker, query: str, hits: list[Hit], texts: list[str]) -> list[Hit]:
    if not hits:
        return []
    scores = reranker.predict([(query, texts[h.chunk_index]) for h in hits], show_progress_bar=False)
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    return sorted(hits, key=lambda h: h.rerank_score, reverse=True)


def retrieve(query: str, encoder, reranker, matrix: np.ndarray, texts: list[str],
             candidates: int = 25, final_k: int = 5) -> list[Hit]:
    qv = embed_texts(encoder, [query])[0]
    hits = dense_search(qv, matrix, candidates)
    if reranker is not None:
        hits = rerank(reranker, query, hits, texts)
    return hits[:final_k]


def is_answerable(hits: list[Hit], min_rerank: float = 0.0, min_dense: float = 0.30) -> bool:
    """Refuse when even the best passage is a weak match (the 'not in my papers' guard).

    The cross-encoder outputs a logit: above 0 means the model thinks the passage is relevant.
    """
    if not hits:
        return False
    best = hits[0]
    if best.rerank_score is not None:
        return best.rerank_score >= min_rerank
    return best.dense_score >= min_dense
