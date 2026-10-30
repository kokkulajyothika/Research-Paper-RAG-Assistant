"""Download arXiv papers, extract text page by page and split it into overlapping chunks.

Papers are downloaded at run time from arXiv (not stored in this repository).
"""
from __future__ import annotations

import io
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from pypdf import PdfReader

ARXIV_PDF = "https://arxiv.org/pdf/{id}"
ARXIV_API = "http://export.arxiv.org/api/query?id_list={ids}"
HEADERS = {"User-Agent": "research-paper-rag/1.0 (portfolio project)"}


@dataclass
class Chunk:
    chunk_id: int
    paper_id: str
    title: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str) -> str:
    """Join hyphenated line breaks and collapse whitespace from PDF extraction."""
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = text.replace("­", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_page(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    """Split one page into chunks of `size` words with `overlap` words shared between neighbours."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        piece = words[start:start + size]
        if len(piece) >= 25 or not chunks:  # skip tiny tails such as page footers
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def fetch_titles(ids: list[str]) -> dict[str, str]:
    """Look up the real title of each paper from the arXiv API (so citations show the true title)."""
    try:
        r = requests.get(ARXIV_API.format(ids=",".join(ids)), headers=HEADERS, timeout=30)
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        out = {}
        for entry in root.findall("a:entry", ns):
            raw_id = entry.findtext("a:id", default="", namespaces=ns).rsplit("/", 1)[-1]
            title = re.sub(r"\s+", " ", entry.findtext("a:title", default="", namespaces=ns)).strip()
            out[re.sub(r"v\d+$", "", raw_id)] = title
        return out
    except Exception:
        return {}


def pages_from_pdf(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [clean_text(p.extract_text() or "") for p in reader.pages]


def build_chunks(papers: list[dict], cache_dir: Path, max_pages: int = 30, log=print) -> list[Chunk]:
    """Download each paper (cached on disk), extract pages and return all chunks."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    titles = fetch_titles([p["id"] for p in papers])
    chunks: list[Chunk] = []
    for p in papers:
        pdf_path = cache_dir / f"{p['id']}.pdf"
        if not pdf_path.exists():
            log(f"Downloading {p['id']} …")
            r = requests.get(ARXIV_PDF.format(id=p["id"]), headers=HEADERS, timeout=60)
            r.raise_for_status()
            pdf_path.write_bytes(r.content)
            time.sleep(1)  # be polite to arXiv
        title = titles.get(p["id"]) or p["short"]
        for page_no, text in enumerate(pages_from_pdf(pdf_path.read_bytes())[:max_pages], start=1):
            for piece in chunk_page(text):
                chunks.append(Chunk(len(chunks), p["id"], title, page_no, piece))
    log(f"Built {len(chunks)} chunks from {len(papers)} papers.")
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.write_text("\n".join(json.dumps(c.to_dict()) for c in chunks), encoding="utf-8")


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
