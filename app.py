"""Streamlit app: ask questions about 15 AI research papers and get verified, cited answers."""
from __future__ import annotations

import html
import os
import time

import streamlit as st

from rag.answer import DEFAULT_MODEL, answer_question
from rag.index import get_index, get_metadata, load_papers
from rag.retrieve import is_answerable, load_models, retrieve

st.set_page_config(page_title="Research Paper Assistant", page_icon="📚", layout="centered")

# Colours used to tag each source card, cycled in order.
TAG_COLOURS = ["#00f0ff", "#ff2bd6", "#f5ff3b", "#39ff88", "#9d4dff", "#ff8a00", "#00f0ff", "#ff2bd6"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root { --cyan:#00f0ff; --pink:#ff2bd6; --yellow:#f5ff3b; --violet:#9d4dff; --ink:#05010d; }
html, body, .stApp p, .stApp li, .stApp label, .stApp button, .stApp summary, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stWidgetLabel"] { font-family:'Inter', sans-serif !important; }
.stApp {
  background:
    linear-gradient(rgba(0,240,255,.05) 1px, transparent 1px) 0 0/44px 44px,
    linear-gradient(90deg, rgba(255,43,214,.05) 1px, transparent 1px) 0 0/44px 44px,
    radial-gradient(900px 500px at 0% 0%, rgba(157,77,255,.30), transparent 60%),
    radial-gradient(800px 500px at 100% 20%, rgba(255,43,214,.22), transparent 60%),
    radial-gradient(900px 600px at 50% 120%, rgba(0,240,255,.18), transparent 60%),
    var(--ink);
}
.stApp::after { content:""; position:fixed; inset:0; pointer-events:none; z-index:999;
  background:repeating-linear-gradient(0deg, rgba(255,255,255,.025) 0 1px, transparent 1px 3px); }
.block-container { padding-top:2.2rem; max-width:880px; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }

.hero { padding:6px 0 10px; }
.badge { display:inline-block; font-family:'JetBrains Mono', monospace !important; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--yellow); border:1px solid var(--yellow); padding:5px 12px; box-shadow:0 0 12px rgba(245,255,59,.35), inset 0 0 8px rgba(245,255,59,.15);
  clip-path:polygon(8px 0,100% 0,100% calc(100% - 8px),calc(100% - 8px) 100%,0 100%,0 8px); }
.hero h1 { font-family:'Space Grotesk', sans-serif !important; font-size:2.8rem; line-height:1.05; font-weight:700; margin:.7rem 0 .5rem;
  text-transform:uppercase; letter-spacing:.04em; color:#fff;
  text-shadow:0 0 6px var(--cyan), 0 0 18px var(--cyan), 0 0 38px rgba(0,240,255,.6), 3px 3px 0 var(--pink); }
.hero h1 span { color:var(--pink); text-shadow:0 0 6px var(--pink), 0 0 20px var(--pink), 0 0 40px rgba(255,43,214,.6), -3px -2px 0 var(--cyan); }
.hero p { color:#c9c3e6; font-size:1.02rem; line-height:1.55; margin:0; }

.pipeline { display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 6px; align-items:center; }
.step { font-family:'JetBrains Mono', monospace !important; font-size:.76rem; color:#e9e4ff; padding:7px 12px;
  background:rgba(157,77,255,.10); border:1px solid rgba(157,77,255,.55); box-shadow:0 0 10px rgba(157,77,255,.25);
  clip-path:polygon(6px 0,100% 0,100% calc(100% - 6px),calc(100% - 6px) 100%,0 100%,0 6px); }
.step b { color:var(--cyan); margin-right:6px; text-shadow:0 0 8px var(--cyan); }
.arrow { color:var(--pink); font-size:.9rem; text-shadow:0 0 8px var(--pink); }

.label { font-family:'JetBrains Mono', monospace !important; font-size:.76rem; letter-spacing:.18em; text-transform:uppercase;
  color:var(--cyan); margin:24px 0 9px; text-shadow:0 0 8px rgba(0,240,255,.7); }
.label::before { content:"// "; color:var(--pink); }

div[data-testid="stButton"] button {
  width:100%; border-radius:0; padding:12px 14px; min-height:64px; font-size:.95rem; font-weight:600;
  background:linear-gradient(135deg, rgba(0,240,255,.07), rgba(255,43,214,.07));
  border:1px solid rgba(0,240,255,.45); color:#e8fbff; transition:all .15s ease;
  clip-path:polygon(12px 0,100% 0,100% calc(100% - 12px),calc(100% - 12px) 100%,0 100%,0 12px); }
div[data-testid="stButton"] button:hover { border-color:var(--pink); color:#fff; background:rgba(255,43,214,.14);
  box-shadow:0 0 18px rgba(255,43,214,.55), inset 0 0 12px rgba(255,43,214,.25); text-shadow:0 0 6px var(--pink); }

div[data-testid="stTextInput"] input { border-radius:0; padding:14px 16px; font-size:1.1rem; font-family:'JetBrains Mono', monospace !important;
  background:rgba(5,1,13,.9); border:1px solid var(--pink); color:var(--cyan); caret-color:var(--yellow);
  box-shadow:0 0 14px rgba(255,43,214,.35), inset 0 0 10px rgba(255,43,214,.12); }
div[data-testid="stTextInput"] input::placeholder { color:#7d6fa8; }
div[data-testid="stTextInput"] input:focus { border-color:var(--cyan); box-shadow:0 0 22px rgba(0,240,255,.55), inset 0 0 12px rgba(0,240,255,.15); }

.card { padding:22px 24px; margin:18px 0 8px; position:relative;
  background:linear-gradient(160deg, rgba(20,6,40,.92), rgba(5,1,13,.92));
  border:1px solid var(--cyan); box-shadow:0 0 22px rgba(0,240,255,.35), inset 0 0 30px rgba(157,77,255,.12);
  clip-path:polygon(18px 0,100% 0,100% calc(100% - 18px),calc(100% - 18px) 100%,0 100%,0 18px); }
.card h3 { font-family:'Space Grotesk', sans-serif !important; margin:0 0 14px; font-size:1rem; letter-spacing:.12em; text-transform:uppercase;
  color:var(--yellow); text-shadow:0 0 10px rgba(245,255,59,.7); }
.claim { margin:0 0 16px; }
.claim p { color:#f3efff; font-size:1.02rem; line-height:1.6; margin:0 0 8px; font-weight:500; }
.cite { display:inline-block; font-family:'JetBrains Mono', monospace !important; font-size:.72rem; font-weight:600; color:#05010d;
  padding:1px 7px; margin-left:6px; vertical-align:2px; box-shadow:0 0 10px currentColor; }
.quote { border-left:3px solid var(--cyan); background:rgba(0,240,255,.06); padding:9px 13px; color:#bfeaff; font-size:.92rem; line-height:1.5;
  box-shadow:inset 0 0 14px rgba(0,240,255,.08); }
.quote .ok { font-family:'JetBrains Mono', monospace !important; color:#39ff88; font-size:.7rem; letter-spacing:.1em; text-transform:uppercase;
  display:block; margin-bottom:4px; text-shadow:0 0 8px rgba(57,255,136,.8); }
.quote i { color:#e8fbff; }

.stats { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
.stat { font-family:'JetBrains Mono', monospace !important; font-size:.74rem; color:#d6cff5; padding:5px 11px;
  border:1px solid rgba(255,43,214,.5); background:rgba(255,43,214,.07); }
.stat b { color:var(--pink); text-shadow:0 0 6px var(--pink); }

.notice { padding:16px 18px; margin:18px 0 8px; font-size:1.05rem; line-height:1.5; font-weight:600;
  background:rgba(245,255,59,.06); border:1px solid var(--yellow); color:var(--yellow); box-shadow:0 0 16px rgba(245,255,59,.25);
  clip-path:polygon(12px 0,100% 0,100% calc(100% - 12px),calc(100% - 12px) 100%,0 100%,0 12px); }
.notice.err { background:rgba(255,43,214,.07); border-color:var(--pink); color:#ffc2f1; box-shadow:0 0 16px rgba(255,43,214,.3); }

div[data-testid="stExpander"] { border-radius:0 !important; border:1px solid rgba(157,77,255,.45) !important;
  background:rgba(12,4,26,.75); margin-bottom:8px; }
div[data-testid="stExpander"] summary { font-weight:600; }
h1, h2, h3, .card h3 { font-family:'Space Grotesk', sans-serif !important; }
code, pre, .stCaption code { font-family:'JetBrains Mono', monospace !important; }
div[data-testid="stExpander"] summary:hover { color:var(--cyan); text-shadow:0 0 8px rgba(0,240,255,.6); }

section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d0420,#05010d); border-right:1px solid rgba(255,43,214,.4);
  box-shadow:4px 0 24px rgba(255,43,214,.12); }
.paper { display:flex; justify-content:space-between; gap:8px; padding:7px 10px; margin:5px 0; font-size:.95rem;
  background:rgba(0,240,255,.04); border-left:2px solid var(--cyan); }
.paper:nth-child(3n+2) { border-left-color:var(--pink); }
.paper:nth-child(3n) { border-left-color:var(--yellow); }
.paper a { color:#e8e2ff !important; text-decoration:none; font-weight:600; }
.paper a:hover { color:var(--cyan) !important; text-shadow:0 0 8px var(--cyan); }
.paper span { color:#8a7fb5; font-family:'JetBrains Mono', monospace !important; font-size:.75rem; }
div[data-baseweb="slider"] div[role="slider"] { background:var(--pink) !important; box-shadow:0 0 12px var(--pink); }
section[data-testid="stSidebar"] div[data-testid="stButton"] { margin-bottom:-6px; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button { min-height:0; padding:9px 12px; clip-path:none; border:none;
  border-left:2px solid rgba(0,240,255,.55); background:rgba(0,240,255,.035); justify-content:flex-start !important; text-align:left; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover { border-left-color:var(--pink); background:rgba(255,43,214,.10);
  box-shadow:none; text-shadow:none; transform:none; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button > div { width:100%; justify-content:flex-start !important; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button > div > span { flex:1 1 auto; width:100%; justify-content:flex-start !important; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button div[data-testid="stMarkdownContainer"] { width:100%; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button p { display:grid; grid-template-columns:1fr auto; column-gap:12px;
  align-items:baseline; width:100%; margin:0; text-align:left !important; font-size:.88rem; font-weight:500; line-height:1.35; color:#e8e2ff; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button p code { font-family:'JetBrains Mono', monospace !important;
  font-size:.72rem; color:#8a7fb5; background:transparent; padding:0; border:none; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover p code { color:var(--pink); }
div[data-testid="stExpander"] div[data-testid="stButton"] button, div[data-testid="stLinkButton"] a { min-height:0; padding:9px 14px; font-size:.88rem; }
div[data-testid="stLinkButton"] a { border-radius:0; background:linear-gradient(135deg, rgba(255,43,214,.18), rgba(157,77,255,.18));
  border:1px solid var(--pink); color:#fff !important; font-weight:600; box-shadow:0 0 14px rgba(255,43,214,.35);
  clip-path:polygon(10px 0,100% 0,100% calc(100% - 10px),calc(100% - 10px) 100%,0 100%,0 10px); }
div[data-testid="stLinkButton"] a:hover { border-color:var(--cyan); box-shadow:0 0 20px rgba(0,240,255,.5); }
div[role="dialog"] { background:linear-gradient(160deg,#160630,#07020f) !important; border:1px solid var(--cyan) !important;
  border-radius:0 !important; box-shadow:0 0 40px rgba(0,240,255,.35), 0 0 80px rgba(255,43,214,.15) !important; }
.pp h2 { font-family:'Space Grotesk', sans-serif !important; font-size:1.7rem; line-height:1.2; margin:.5rem 0 .4rem; color:#fff;
  text-shadow:0 0 14px rgba(0,240,255,.45); }
.pp-tag { font-family:'JetBrains Mono', monospace !important; font-size:.72rem; letter-spacing:.1em; color:var(--yellow);
  border:1px solid var(--yellow); padding:3px 9px; }
.pp-auth { color:#b8acdf; font-size:.92rem; margin-bottom:6px; }
.pp-h { font-family:'JetBrains Mono', monospace !important; color:var(--cyan); font-size:.74rem; letter-spacing:.16em;
  text-transform:uppercase; margin:16px 0 6px; }
.pp-abs { color:#e3dcff; line-height:1.65; font-size:.97rem; }
.foot { text-align:center; color:#7d6fa8; font-family:'JetBrains Mono', monospace !important; font-size:.76rem; letter-spacing:.06em; margin:40px 0 10px; }
.foot a { color:var(--cyan); text-decoration:none; text-shadow:0 0 6px rgba(0,240,255,.6); }
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


# ---------- paper panel ----------
PAPERS = load_papers()
BY_ID = {p["id"]: p for p in PAPERS}


@st.cache_data(show_spinner=False)
def paper_meta() -> dict:
    return get_metadata()


@st.dialog("Paper details", width="large")
def show_paper(pid: str, page: int | None = None):
    p, m = BY_ID[pid], paper_meta().get(pid, {})
    authors = m.get("authors", [])
    who = ", ".join(authors[:6]) + (f" and {len(authors) - 6} more" if len(authors) > 6 else "")
    st.markdown(
        f"<div class='pp'><span class='pp-tag'>arXiv {esc(pid)} · {esc(m.get('published') or p['year'])}</span>"
        f"<h2>{esc(m.get('title') or p['short'])}</h2>"
        + (f"<div class='pp-auth'>{esc(who)}</div>" if who else "")
        + (f"<div class='pp-h'>// Abstract</div><p class='pp-abs'>{esc(m['summary'])}</p>" if m.get("summary") else
           "<p class='pp-abs'>Details are unavailable right now; you can still open the PDF.</p>")
        + "</div>", unsafe_allow_html=True)
    url = f"https://arxiv.org/pdf/{pid}" + (f"#page={page}" if page else "")
    st.link_button(f"Read the PDF{f' at page {page}' if page else ''} ↗", url, use_container_width=True)


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("<div class='label' style='margin-top:4px'>Papers in the index</div>", unsafe_allow_html=True)
    for p in PAPERS:
        if st.button(f"{p['short']} `{p['year']}`", key=f"paper-{p['id']}", use_container_width=True):
            show_paper(p["id"])
    st.markdown("<div class='label'>Settings</div>", unsafe_allow_html=True)
    final_k = st.slider("Passages given to the model", 3, 8, 5)
    show_scores = st.toggle("Show retrieval scores", value=False)
    st.caption(f"Answer model: `{DEFAULT_MODEL}` via Groq")

# ---------- hero ----------
st.markdown("""
<div class="hero">
  <span class="badge">&gt; RAG · PyTorch · Verified citations_</span>
  <h1>Research Paper <span>Assistant</span></h1>
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
            st.markdown(f"<div class='card'><h3>▶ Answer</h3>{''.join(parts)}{stats}</div>", unsafe_allow_html=True)
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
            c1, c2 = st.columns(2)
            c1.link_button(f"Open PDF at page {ch.page} ↗", f"https://arxiv.org/pdf/{ch.paper_id}#page={ch.page}",
                           use_container_width=True)
            if c2.button("About this paper", key=f"about-{i}", use_container_width=True):
                show_paper(ch.paper_id, ch.page)

st.markdown("<div class='foot'>Built by <a href='https://www.linkedin.com/in/jyothikakokkula' target='_blank'>Jyothika Kokkula</a> · "
            "<a href='https://github.com/kokkulajyothika/Research-Paper-RAG-Assistant' target='_blank'>Source on GitHub</a></div>",
            unsafe_allow_html=True)
