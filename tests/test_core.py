"""Unit tests for the parts that don't need downloaded models (run with: pytest -q)."""
import json

import numpy as np
import pytest

from eval.metrics import recall_at_k, reciprocal_rank, refusal_accuracy, summarise
from rag.answer import Answer, answer_question, parse_and_verify, quote_in_source
from rag.ingest import chunk_page, clean_text
from rag.retrieve import Hit, dense_search, is_answerable, rerank, retrieve


# ---------- ingest ----------
def test_clean_text_joins_hyphenation_and_spaces():
    assert clean_text("trans-\nformer   model\n\nworks") == "transformer model works"


def test_chunk_page_overlap_and_coverage():
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_page(" ".join(words), size=180, overlap=40)
    assert len(chunks) == 3
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]  # overlap kept
    assert chunks[-1].split()[-1] == "w399"                     # nothing lost at the end


def test_chunk_page_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_page("a b c", size=10, overlap=10)


# ---------- retrieval ----------
class FakeEncoder:
    """Bag-of-words 'embedding' over a tiny vocabulary, L2-normalised like the real model."""
    vocab = ["attention", "retrieval", "agent", "bread", "memory", "token"]

    def encode(self, texts, **_):
        out = []
        for t in texts:
            v = np.array([t.lower().count(w) for w in self.vocab], dtype=np.float32) + 1e-6
            out.append(v / np.linalg.norm(v))
        return np.array(out)


class FakeReranker:
    def predict(self, pairs, **_):
        return [3.0 if q.split()[0].lower() in p.lower() else -3.0 for q, p in pairs]


TEXTS = ["attention attention token", "retrieval memory retrieval", "agent agent tools", "bread flour oven"]


def test_dense_search_returns_best_match_first():
    m = FakeEncoder().encode(TEXTS)
    hits = dense_search(FakeEncoder().encode(["retrieval memory"])[0], m, 2)
    assert hits[0].chunk_index == 1 and len(hits) == 2
    assert hits[0].dense_score >= hits[1].dense_score


def test_rerank_reorders_by_cross_encoder_score():
    hits = [Hit(0, 0.9), Hit(2, 0.5)]
    out = rerank(FakeReranker(), "agent loop", hits, TEXTS)
    assert out[0].chunk_index == 2 and out[0].rerank_score == 3.0


def test_retrieve_end_to_end_with_fakes():
    m = FakeEncoder().encode(TEXTS)
    hits = retrieve("attention token", FakeEncoder(), FakeReranker(), m, TEXTS, candidates=3, final_k=2)
    assert hits[0].chunk_index == 0


def test_is_answerable_threshold():
    assert is_answerable([Hit(0, 0.9, 2.0)])
    assert not is_answerable([Hit(0, 0.9, -1.5)])
    assert not is_answerable([])
    assert is_answerable([Hit(0, 0.45)]) and not is_answerable([Hit(0, 0.1)])


# ---------- citation verification ----------
SOURCES = ["The Transformer relies entirely on an attention mechanism to draw global dependencies between input and output.",
           "RAG models combine parametric and non-parametric memory for language generation."]


def test_quote_in_source_ignores_case_and_punctuation():
    assert quote_in_source("relies entirely on an Attention mechanism", SOURCES[0])
    assert not quote_in_source("relies entirely on recurrence", SOURCES[0])
    assert not quote_in_source("attention", SOURCES[0])  # too short to count as evidence


def test_parse_and_verify_keeps_only_supported_claims():
    raw = json.dumps({"answerable": True, "claims": [
        {"text": "Transformers use attention.", "source": 1, "quote": "relies entirely on an attention mechanism"},
        {"text": "RAG uses a knowledge graph.", "source": 2, "quote": "uses a large knowledge graph"},
        {"text": "Bad source number.", "source": 9, "quote": "combine parametric and non-parametric memory"}]})
    ans = parse_and_verify(raw, SOURCES)
    assert [c.source for c in ans.claims] == [1]
    assert len(ans.removed) == 2 and ans.answerable
    assert abs(ans.verified_ratio - 1 / 3) < 1e-9


def test_parse_and_verify_refuses_when_nothing_verified():
    raw = json.dumps({"answerable": True, "claims": [{"text": "x", "source": 1, "quote": "something not in the text at all"}]})
    assert not parse_and_verify(raw, SOURCES).answerable


def test_answer_question_without_key_does_not_call_llm():
    called = []
    ans = answer_question("q", SOURCES, None, llm=lambda *a: called.append(1))
    assert ans.error and not called


def test_answer_question_handles_llm_failure():
    def boom(*a):
        raise RuntimeError("429 rate limit")
    ans = answer_question("q", SOURCES, "key", llm=boom)
    assert isinstance(ans, Answer) and "429" in ans.error


def test_quote_matching_handles_pdf_quirks():
    pdf = "This yields a signi\ufb01cant improvement because self- attention connects all positions."
    assert quote_in_source("yields a significant improvement because self-attention", pdf)
    # one word dropped by PDF extraction is tolerated in a long quote
    long_src = "the number of operations required to relate signals from two arbitrary positions grows linearly"
    assert quote_in_source("number of operations required to relate the signals from two arbitrary positions grows", long_src)
    # a different statement is still rejected
    assert not quote_in_source("the number of operations grows exponentially with distance always", long_src)


def test_answer_question_retries_when_quote_not_exact():
    calls = []
    def llm(q, *a):
        calls.append(q)
        quote = "relies on an attention mechanism to draw dependencies" if len(calls) == 1 else "relies entirely on an attention mechanism"
        return json.dumps({"answerable": True, "claims": [{"text": "Uses attention.", "source": 1, "quote": quote}]})
    ans = answer_question("q", SOURCES, "key", llm=llm)
    assert len(calls) == 2 and "NOT found word-for-word" in calls[1]
    assert ans.answerable and len(ans.claims) == 1 and not ans.removed


# ---------- metrics ----------
def test_metrics():
    assert recall_at_k(["a", "b", "c"], "b", 1) == 0.0 and recall_at_k(["a", "b"], "b", 2) == 1.0
    assert reciprocal_rank(["a", "b", "c"], "c") == pytest.approx(1 / 3)
    assert refusal_accuracy([True, False], [True, True]) == 0.5
    s = summarise([{"ranked": ["x", "y"], "gold": "y", "pred_answerable": True, "answerable": True},
                   {"ranked": [], "gold": None, "pred_answerable": False, "answerable": False}])
    assert s["recall@1"] == 0.0 and s["recall@3"] == 1.0 and s["mrr"] == 0.5 and s["refusal_accuracy"] == 1.0
