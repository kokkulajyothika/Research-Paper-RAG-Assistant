"""Streamlit app: ask questions about 15 AI research papers and get verified, cited answers."""
from __future__ import annotations

import html
import os
import time

import streamlit as st

from rag.answer import DEFAULT_MODEL, answer_question
from rag.index import get_index, load_papers
from rag.retrieve import is_answerable, load_models, retrieve

st.set_page_config(page_title="Research Paper Assistant", page_icon="📚", layout="centered")

# Colours used to tag each source card, cycled in order.
TAG_COLOURS = ["#818cf8", "#22d3ee", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb7185", "#38bdf8"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput input, button { font-family: 'Inter', sans-serif !important; }
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,.22), transparent 60%),
    radial-gradient(900px 500px at 110% 10%, rgba(236,72,153,.16), transparent 60%),
    radial-gradient(900px 600px at 50% 120%, rgba(34,211,238,.14), transparent 60%),
    #070b1d;
}
.block-container { padding-top: 2.2rem; max-width: 860px; }
#MainMenu, footer { visibility: hidden; }

.hero { padding: 6px 0 10px; }
.badge { display:inline-block; font-size:.72rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
  color:#c7d2fe; background:rgba(99,102,241,.16); border:1px solid rgba(129,140,248,.35);
  padding:5px 12px; border-radius:999px; }
.hero h1 { font-size:2.7rem; line-height:1.1; font-weight:800; margin:.6rem 0 .5rem; letter-spacing:-.02em;
  background: linear-gradient(90deg,#a5b4fc 0%,#67e8f9 45%,#f9a8d4 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { color:#a6b0cf; font-size:1.02rem; line-height:1.6; margin:0; }

.pipeline { display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 6px; align-items:center; }
.step { font-size:.78rem; font-weight:500; color:#e2e8f0; padding:6px 11px; border-radius:10px;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09); }
.step b { color:#67e8f9; font-weight:700; margin-right:4px; }
.arrow { color:#475569; font-size:.8rem; }

.label { font-size:.78rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#8b95b8; margin:22px 0 8px; }

div[data-testid="stButton"] button {
  width:100%; text-align:left; justify-content:flex-start; border-radius:14px; padding:12px 14px; min-height:64px;
  background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));
  border:1px solid rgba(255,255,255,.10); color:#dbe4ff; font-weight:500; transition:all .18s ease; }
div[data-testid="stButton"] button:hover { border-color:#818cf8; color:#fff; transform:translateY(-1px);
  box-shadow:0 8px 24px rgba(99,102,241,.25); }

div[data-testid="stTextInput"] input { border-radius:14px; padding:14px 16px; font-size:1rem;
  background:rgba(15,23,42,.85); border:1px solid rgba(129,140,248,.35); color:#f1f5f9; }
div[data-testid="stTextInput"] input:focus { border-color:#22d3ee; box-shadow:0 0 0 3px rgba(34,211,238,.18); }

.card { border-radius:18px; padding:22px 24px; margin:18px 0 8px;
  background:linear-gradient(180deg,rgba(30,41,82,.72),rgba(15,23,42,.72));
  border:1px solid rgba(129,140,248,.28); box-shadow:0 18px 50px rgba(2,6,23,.55); }
.card h3 { margin:0 0 14px; font-size:1.05rem; font-weight:700; color:#e0e7ff; display:flex; align-items:center; gap:8px; }
.claim { margin:0 0 16px; }
.claim p { color:#f1f5f9; font-size:1.02rem; line-height:1.65; margin:0 0 8px; }
.cite { display:inline-block; font-size:.72rem; font-weight:700; color:#0b1020; padding:2px 7px; border-radius:6px; margin-left:4px; vertical-align:2px; }
.quote { border-left:3px solid #34d399; background:rgba(52,211,153,.07); border-radius:0 10px 10px 0;
  padding:9px 13px; color:#b6c2dc; font-size:.9rem; line-height:1.55; }
.quote .ok { color:#34d399; font-weight:700; font-size:.75rem; letter-spacing:.04em; text-transform:uppercase; display:block; margin-bottom:3px; }
.quote i { color:#e2e8f0; }

.stats { display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }
.stat { font-size:.78rem; color:#cbd5e1; padding:5px 11px; border-radius:999px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08); }
.stat b { color:#fff; }

.notice { border-radius:16px; padding:16px 18px; margin:18px 0 8px; font-size:.97rem; line-height:1.55;
  background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.35); color:#fde68a; }
.notice.err { background:rgba(244,63,94,.08); border-color:rgba(244,63,94,.4); color:#fecdd3; }

.src-head { display:flex; align-items:center; gap:10px; }
div[data-testid="stExpander"] { border-radius:14px !important; border:1px solid rgba(255,255,255,.09) !important;
  background:rgba(15,23,42,.55); margin-bottom:8px; }
div[data-testid="stExpander"] summary:hover { color:#a5b4fc; }

section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1130,#0a0f24); border-right:1px solid rgba(255,255,255,.06); }
.paper { display:flex; justify-content:space-between; gap:8px; padding:7px 10px; margin:4px 0; border-radius:10px;
  background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.06); font-size:.84rem; }
.paper a { color:#dbe4ff !important; text-decoration:none; }
.paper a:hover { color:#67e8f9 !important; }
.paper span { color:#64748b; font-size:.78rem; }
.foot { text-align:center; color:#64748b; font-size:.8rem; margin:36px 0 10px; }
.foot a { color:#a5b4fc; text-decoration:none; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading PyTorch models and building the index (first run takes a few minutes)…")
def setup():
    encoder, reranker = load_models()
    chunks, matrix = get_index(encoder, log=print)
    return encoder, reranker, chunks, matrix


def get_api_key() -> str | None:
    try:
        return st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except Exception:
        return os.getenv("GROQ_API_KEY")


def esc(text: str) -> str:
    return html.escape(str(text))


def colour(i: int) -> str:
    return TAG_COLOURS[(i - 1) % len(TAG_COLOURS)]


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("<div class='label' style='margin-top:4px'>Papers in the index</div>", unsafe_allow_html=True)
    st.markdown("".join(
        f"<div class='paper'><a href='https://arxiv.org/abs/{esc(p['id'])}' target='_blank'>{esc(p['short'])}</a>"
        f"<span>{esc(p['year'])}</span></div>" for p in load_papers()), unsafe_allow_html=True)
    st.markdown("<div class='label'>Settings</div>", unsafe_allow_html=True)
    final_k = st.slider("Passages given to the model", 3, 8, 5)
    show_scores = st.toggle("Show retrieval scores", value=False)
    st.caption(f"Answer model: `{DEFAULT_MODEL}` via Groq")

# ---------- hero ----------
st.markdown("""
<div class="hero">
  <span class="badge">RAG · PyTorch · Verified citations</span>
  <h1>Research Paper Assistant</h1>
  <p>Ask anything about 15 landmark AI papers, from Transformers to Self-RAG. Every sentence in the answer
  cites a paper and page, and every quote is checked against the source text. If the papers don't say it, the assistant won't either.</p>
  <div class="pipeline">
    <span class="step"><b>1</b>Bi-encoder search</span><span class="arrow">→</span>
    <span class="step"><b>2</b>Cross-encoder rerank</span><span class="arrow">→</span>
    <span class="step"><b>3</b>Cited LLM answer</span><span class="arrow">→</span>
    <span class="step"><b>4</b>Quote verification</span>
  </div>
</div>
""", unsafe_allow_html=True)

encoder, reranker, chunks, matrix = setup()
texts = [c.text for c in chunks]

examples = ["What problem does self-attention solve compared with recurrent networks?",
            "How does Self-RAG decide when to retrieve?",
            "What does 'lost in the middle' mean for long contexts?",
            "How does ReAct combine reasoning and acting?"]
icons = ["⚡", "🔁", "🧭", "📏"]

st.markdown("<div class='label'>Try an example</div>", unsafe_allow_html=True)
row1, row2 = st.columns(2), st.columns(2)
for col, ex, ic in zip([*row1, *row2], examples, icons):
    if col.button(f"{ic}  {ex}", use_container_width=True):
        st.session_state["q"] = ex

st.markdown("<div class='label'>Or ask your own</div>", unsafe_allow_html=True)
question = st.text_input("Your question", key="q", label_visibility="collapsed",
                         placeholder="e.g. What is dense passage retrieval?")

if question:
    t0 = time.time()
    with st.spinner("Searching the papers…"):
        hits = retrieve(question, encoder, reranker, matrix, texts, candidates=25, final_k=final_k)
    sources = [texts[h.chunk_index] for h in hits]

    if not is_answerable(hits):
        st.markdown("<div class='notice'>🔎 I couldn't find this in my 15 papers, so I won't guess. "
                    "Try rephrasing, or ask about a topic they cover.</div>", unsafe_allow_html=True)
    else:
        with st.spinner("Writing a cited answer…"):
            ans = answer_question(question, sources, get_api_key())
        elapsed = time.time() - t0
        if ans.error:
            st.markdown(f"<div class='notice err'>⚠️ {esc(ans.error)}</div>", unsafe_allow_html=True)
        elif not ans.answerable:
            st.markdown("<div class='notice'>🤔 The retrieved passages don't contain a clear answer, so I won't guess.</div>",
                        unsafe_allow_html=True)
        else:
            parts = []
            for c in ans.claims:
                ch = chunks[hits[c.source - 1].chunk_index]
                parts.append(
                    f"<div class='claim'><p>{esc(c.text)}<span class='cite' style='background:{colour(c.source)}'>"
                    f"{c.source}</span></p><div class='quote'><span class='ok'>✓ Verified quote · {esc(ch.title)}, p.{ch.page}</span>"
                    f"<i>“{esc(c.quote)}”</i></div></div>")
            n_ok, n_all = len(ans.claims), len(ans.claims) + len(ans.removed)
            stats = (f"<div class='stats'><span class='stat'>⏱ <b>{elapsed:.1f}s</b></span>"
                     f"<span class='stat'>✅ <b>{n_ok}/{n_all}</b> claims verified</span>"
                     f"<span class='stat'>📄 <b>{len(hits)}</b> passages read</span></div>")
            st.markdown(f"<div class='card'><h3>✨ Answer</h3>{''.join(parts)}{stats}</div>", unsafe_allow_html=True)
        if not ans.error and ans.removed:
            with st.expander(f"⚠️ {len(ans.removed)} claim(s) removed because the quote was not found in the source"):
                for c in ans.removed:
                    st.markdown(f"- {c.text} (cited [{c.source}]: “{c.quote}”)")

    st.markdown("<div class='label'>Sources</div>", unsafe_allow_html=True)
    for i, h in enumerate(hits, start=1):
        ch = chunks[h.chunk_index]
        label = f"[{i}]  {ch.title}  ·  page {ch.page}"
        if show_scores:
            label += f"   (dense {h.dense_score:.2f} · rerank {h.rerank_score:.2f})"
        with st.expander(label):
            st.markdown(f"<div style='border-left:3px solid {colour(i)};padding-left:12px;color:#cbd5e1;line-height:1.6'>"
                        f"{esc(ch.text)}</div>", unsafe_allow_html=True)
            st.markdown(f"[Open paper on arXiv ↗](https://arxiv.org/abs/{ch.paper_id})")

st.markdown("<div class='foot'>Built by <a href='https://www.linkedin.com/in/jyothikakokkula' target='_blank'>Jyothika Kokkula</a> · "
            "<a href='https://github.com/kokkulajyothika/Research-Paper-RAG-Assistant' target='_blank'>Source on GitHub</a></div>",
            unsafe_allow_html=True)
