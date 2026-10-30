"""Generate a cited answer with the Groq API, then verify every quote in code.

The LLM must return JSON. Each claim cites a source number and an exact quote from that source.
Code then checks that the quote really appears in the cited passage; unverified citations are
removed and reported, so the user can see exactly what is supported by the papers.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """You answer questions about AI research papers using ONLY the numbered sources provided.
Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Every claim must cite one source number and include a short EXACT quote (5-25 words) copied from that source.
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
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9%.,'\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_source(quote: str, source_text: str, min_words: int = 4) -> bool:
    """True if the quote appears in the source after normalising spacing, case and punctuation."""
    q = normalise(quote)
    if len(q.split()) < min_words:
        return False
    return q in normalise(source_text)


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
    r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def answer_question(question: str, sources: list[str], api_key: str | None, model: str = DEFAULT_MODEL,
                    llm=call_groq) -> Answer:
    if not api_key:
        return Answer(False, error="No GROQ_API_KEY set. Add it as a secret to generate answers.")
    try:
        return parse_and_verify(llm(question, sources, api_key, model), sources)
    except Exception as e:  # network, rate limit or bad JSON
        return Answer(False, error=f"Answer generation failed: {str(e)[:200]}")
