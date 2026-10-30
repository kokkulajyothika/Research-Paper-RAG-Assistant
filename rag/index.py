"""Build or load the search index (chunks + embedding matrix) on disk."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ingest import build_chunks, fetch_metadata, load_chunks, save_chunks
from .retrieve import embed_texts

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PAPERS_FILE = ROOT / "papers.json"


def load_papers() -> list[dict]:
    return json.loads(PAPERS_FILE.read_text(encoding="utf-8"))


def get_index(encoder, log=print):
    """Return (chunks, matrix). Builds and caches them in ./data on the first run."""
    chunks_path, matrix_path = DATA / "chunks.jsonl", DATA / "embeddings.npy"
    if chunks_path.exists() and matrix_path.exists():
        chunks, matrix = load_chunks(chunks_path), np.load(matrix_path)
        if len(chunks) == len(matrix):
            return chunks, matrix
    DATA.mkdir(exist_ok=True)
    chunks = build_chunks(load_papers(), DATA / "pdfs", log=log)
    log("Embedding chunks with the PyTorch bi-encoder …")
    matrix = embed_texts(encoder, [c.text for c in chunks])
    save_chunks(chunks, chunks_path)
    np.save(matrix_path, matrix)
    return chunks, matrix


def get_metadata() -> dict[str, dict]:
    """Paper details for the app (cached in ./data/meta.json after the first successful fetch)."""
    path = DATA / "meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    meta = fetch_metadata([p["id"] for p in load_papers()])
    if meta:
        DATA.mkdir(exist_ok=True)
        path.write_text(json.dumps(meta), encoding="utf-8")
    return meta
