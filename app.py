"""Streamlit app: ask questions about 15 AI research papers and get verified, cited answers."""
from __future__ import annotations

import os
import time

import streamlit as st

from rag.answer import DEFAULT_MODEL, answer_question
from rag.index import get_index, load_papers
from rag.retrieve import is_answerable, load_models, retrieve

st.set_page_config(page_title="Research Paper Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading PyTorch models and building the index (first run takes a few minutes)…")
def setup():
    encoder, reranker = load_models()
    chunks, matrix = get_index(encoder, log=lambda m: None)
    return encoder, reranker, chunks, matrix


def get_api_key() -> str | None:
    try:
        return st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except Exception:
        return os.getenv("GROQ_API_KEY")


st.title("📚 Research Paper Assistant")
st.caption("Ask about 15 influential AI papers (Transformers, RAG, ReAct, Self-RAG …). "
           "Every answer cites the paper and page, and every quote is checked against the source text.")

with st.sidebar:
    st.header("Papers in the index")
    for p in load_papers():
        st.markdown(f"- [{p['short']}](https://arxiv.org/abs/{p['id']}) ({p['year']})")
    st.divider()
    final_k = st.slider("Passages given to the model", 3, 8, 5)
    show_scores = st.checkbox("Show retrieval scores", value=False)
    st.caption(f"Answer model: `{DEFAULT_MODEL}` via Groq")

encoder, reranker, chunks, matrix = setup()
texts = [c.text for c in chunks]

examples = ["What problem does self-attention solve compared with recurrent networks?",
            "How does Self-RAG decide when to retrieve?",
            "What does 'lost in the middle' mean for long contexts?",
            "What metrics does RAGAS use to evaluate RAG systems?"]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["q"] = ex

question = st.text_input("Your question", key="q", placeholder="e.g. How does ReAct combine reasoning and acting?")

if question:
    t0 = time.time()
    hits = retrieve(question, encoder, reranker, matrix, texts, candidates=25, final_k=final_k)
    sources = [texts[h.chunk_index] for h in hits]

    if not is_answerable(hits):
        st.warning("I couldn't find this in my 15 papers, so I won't guess. Try rephrasing, or ask about a topic they cover.")
    else:
        with st.spinner("Writing a cited answer…"):
            ans = answer_question(question, sources, get_api_key())
        if ans.error:
            st.error(ans.error)
        elif not ans.answerable:
            st.warning("The retrieved passages don't contain a clear answer, so I won't guess.")
        else:
            st.subheader("Answer")
            for c in ans.claims:
                ch = chunks[hits[c.source - 1].chunk_index]
                st.markdown(f"{c.text} **[{c.source}]**")
                st.caption(f"✅ Verified quote from *{ch.title}*, p.{ch.page}: “{c.quote}”")
            if ans.removed:
                with st.expander(f"⚠️ {len(ans.removed)} claim(s) removed because the quote was not found in the source"):
                    for c in ans.removed:
                        st.markdown(f"- {c.text} (cited [{c.source}]: “{c.quote}”)")
        st.caption(f"Answered in {time.time() - t0:.1f}s")

    st.subheader("Sources")
    for i, h in enumerate(hits, start=1):
        ch = chunks[h.chunk_index]
        label = f"[{i}] {ch.title} — page {ch.page}"
        if show_scores:
            label += f"  (dense {h.dense_score:.2f}, rerank {h.rerank_score:.2f})"
        with st.expander(label):
            st.write(ch.text)
            st.markdown(f"[Open paper on arXiv](https://arxiv.org/abs/{ch.paper_id})")
