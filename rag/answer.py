"""Generate a cited answer with the Groq API, then verify every quote in code.

The LLM must return JSON. Each claim cites a source number and an exact quote from that source.
Code then checks that the quote really appears in the cited passage; unverified citations are
removed and reported, so the user can see exactly what is supported by the papers.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from dataclasses import dataclass, field

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """You answer questions about AI research papers using ONLY the numbered sources provided.
Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Every claim must cite one source number and include an EXACT quote (6-25 words): a continuous span copied character-for-character from that source. Never shorten, merge or reword a quote.
- If the sources do not contain the answer, set "answerable" to false and leave "claims" empty.
- Write in clear, plain English.
Return JSON only, in this shape:
{"answerable": true, "claims": [{"text": "one sentence answer", "source": 1, "quote": "exact words from source 1"}]}"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answerable", "claims"],
    "properties": {
        "answerable": {"type": "boolean"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "source", "quote"],
                "properties": {
                    "text": {"type": "string"},
                    "source": {"type": "integer"},
                    "quote": {"type": "string"},
                },
            },
        },
    },
}


@dataclass
class Claim:
    text: str
    source: int
    quote: str
    verified: bool = False


@dataclass
class Answer:
    answerable: bool
    claims: list[Claim] = field(default_factory=list)
    removed: list[Claim] = field(default_factory=list)
    error: str | None = None

    @property
    def verified_ratio(self) -> float:
        total = len(self.claims) + len(self.removed)
        return len(self.claims) / total if total else 0.0


def normalise(text: str) -> str:
    """Lower-case and simplify text so PDF quirks don't break matching.

    NFKC turns PDF ligatures like the single character 'ﬁ' into 'fi'; hyphens and dashes become
    spaces so 'self-attention', 'self- attention' and 'self attention' all match.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[-\u2010-\u2015]", " ", text)
    text = re.sub(r"[^a-z0-9%' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_source(quote: str, source_text: str, min_words: int = 4, min_ratio: float = 0.9) -> bool:
    """True if the quote appears in the source.

    First an exact match after normalising; if that fails, a near-exact match: some window of the
    source with the same number of words must agree with the quote word-for-word at least 90%
    (allows a missed word from PDF extraction, never a different sentence).
    """
    q = normalise(quote).split()
    if len(q) < min_words:
        return False
    src = normalise(source_text).split()
    if " ".join(q) in " ".join(src):
        return True
    n = len(q)
    for start in range(0, max(1, len(src) - n + 1)):
        window = src[start:start + n + 1]
        if SequenceMatcher(None, q, window, autojunk=False).ratio() >= min_ratio:
            return True
    return False


def build_user_prompt(question: str, sources: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
    return f"SOURCES:\n{numbered}\n\nQUESTION: {question}"


def parse_and_verify(raw_json: str, sources: list[str]) -> Answer:
    data = json.loads(raw_json)
    answer = Answer(answerable=bool(data.get("answerable")))
    for c in data.get("claims", []) or []:
        claim = Claim(str(c.get("text", "")), int(c.get("source", 0) or 0), str(c.get("quote", "")))
        ok = 1 <= claim.source <= len(sources) and quote_in_source(claim.quote, sources[claim.source - 1])
        claim.verified = ok
        (answer.claims if ok else answer.removed).append(claim)
    if answer.answerable and not answer.claims:
        answer.answerable = False  # nothing survived verification -> do not present an answer
    return answer


def call_groq(question: str, sources: list[str], api_key: str, model: str = DEFAULT_MODEL,
              timeout: int = 60) -> str:
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, sources)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "cited_answer", "strict": True, "schema": SCHEMA}},
    }
    for attempt in range(4):  # wait and retry when rate-limited (429) or the server is busy (5xx)
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=timeout)
        if r.status_code == 429 or r.status_code >= 500:
            if attempt < 3:
                try:
                    wait = float(r.headers.get("retry-after", 0))
                except ValueError:
                    wait = 0
                time.sleep(min(max(wait, 2 ** attempt * 2), 30))
                continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def answer_question(question: str, sources: list[str], api_key: str | None, model: str = DEFAULT_MODEL,
                    llm=call_groq, retries: int = 1) -> Answer:
    """Ask the LLM, verify its quotes, and if any quote was not exact, retry once with feedback."""
    if not api_key:
        return Answer(False, error="No GROQ_API_KEY set. Add it as a secret to generate answers.")
    try:
        best = parse_and_verify(llm(question, sources, api_key, model), sources)
        for _ in range(retries):
            if not best.removed:
                break
            bad = "\n".join(f'- [{c.source}] "{c.quote}"' for c in best.removed)
            retry_q = (f"{question}\n\nNOTE: in a previous attempt these quotes were NOT found word-for-word in the "
                       f"cited source:\n{bad}\nCopy every quote exactly as a continuous span from the source text.")
            again = parse_and_verify(llm(retry_q, sources, api_key, model), sources)
            if len(again.claims) > len(best.claims) or (len(again.claims) == len(best.claims) and len(again.removed) < len(best.removed)):
                best = again
        return best
    except Exception as e:  # network, rate limit or bad JSON
        return Answer(False, error=f"Answer generation failed: {str(e)[:200]}")
