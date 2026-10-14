"""Classify a query into one of the eval categories.

Keyword rules first (small, readable, free). Optionally, a cheap LLM classifies queries the
rules cannot place. Queries nothing matches are "general" and are routed on overall scores.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from routewise.evals.framework import CATEGORIES
from routewise.models.client import ChatResult, chat

GENERAL = "general"

# (category, regex, weight). Weight 2 marks a strong signal.
RULES: list[tuple[str, str, float]] = [
    ("structured_output", r"\bjson\b", 2),
    ("structured_output", r"\bschema\b", 1),
    ("structured_output", r"\bextract\b", 1),
    ("structured_output", r"\bfields?\b|\bkeys\b", 1),
    ("structured_output", r"\btable\b|\bcsv\b", 1),
    ("structured_output", r"function[- ]call|tool call|\barguments\b", 2),
    ("structured_output", r"\bbullet", 1),
    ("structured_output", r"\bclassify\b|\bcategori[sz]e\b|\btriage\b", 1),
    ("reasoning", r"\bhow many\b|\bhow much\b", 1),
    ("reasoning", r"\bcalculate\b|\bcompute\b|\bsolve\b", 2),
    ("reasoning", r"\bpuzzle\b|\bconstraints?\b|\bschedule\b", 1),
    ("reasoning", r"\bbug\b|\bdebug\b|line number", 2),
    ("reasoning", r"\bflaw\b|\bweakness\b|\bwhat'?s wrong\b", 2),
    ("reasoning", r"\bstep by step\b|\bprove\b", 1),
    ("summarization", r"\bsummar", 2),
    ("summarization", r"\btl;?dr\b|\brecap\b|\bkey points\b", 2),
    ("summarization", r"\baction items?\b|\bmeeting notes\b", 2),
    ("summarization", r"\bcondense\b|\bshorten\b", 1),
    ("voice", r"\btranscript\b|\basr\b|\bspeech\b", 2),
    ("voice", r"\buh\b|\bum+\b", 1),
    ("voice", r"barge[- ]in|\binterrupt", 2),
    ("voice", r"\bspoken\b|\bvoice\b|\bphone call\b|\bcaller\b", 1),
    ("voice", r"\bangry\b|\bupset\b|\bfrustrated\b", 1),
    ("voice", r"\bintent\b|\bslots?\b", 1),
    ("safety", r"ignore (all )?(previous|prior|above) instructions", 3),
    ("safety", r"\bsystem prompt\b|\bjailbreak\b", 2),
    ("safety", r"\bredact\b|\bpii\b|\bpersonal data\b", 2),
    ("safety", r"[\w.+-]+@[\w-]+\.\w+", 1),
    ("safety", r"\bpassword\b|\bsecret\b|\bapi key\b|\bconfidential\b", 1),
]

_COMPILED = [(cat, re.compile(pattern, re.I), weight) for cat, pattern, weight in RULES]


@dataclass
class Classification:
    category: str
    confidence: float
    method: str  # rules | llm | default
    matched: list[str] = field(default_factory=list)


def classify(query: str) -> Classification:
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for category, regex, weight in _COMPILED:
        m = regex.search(query)
        if m:
            scores[category] = scores.get(category, 0.0) + weight
            matched.setdefault(category, []).append(m.group(0))
    if not scores:
        return Classification(GENERAL, 0.0, "default")
    best = max(scores, key=lambda c: (scores[c], -CATEGORIES.index(c)))
    return Classification(best, scores[best] / sum(scores.values()), "rules", matched[best])


LLM_PROMPT = (
    "Classify the user's request into exactly one category: "
    + ", ".join(CATEGORIES)
    + ". If none fits, answer general. Answer with the category name only.\n\nRequest:\n"
)


async def classify_async(
    query: str,
    *,
    use_llm: bool = False,
    llm_model: str = "claude-haiku-5-5",
    chat_fn: Callable[..., Awaitable[ChatResult]] = chat,
) -> Classification:
    """Rules first; when they miss and `use_llm` is set, ask a cheap model."""
    result = classify(query)
    if result.method == "rules" or not use_llm:
        return result
    reply = await chat_fn(
        llm_model, [{"role": "user", "content": LLM_PROMPT + query}], max_tokens=20
    )
    answer = reply.text.strip().lower()
    for category in CATEGORIES:
        if category in answer:
            return Classification(category, 0.5, "llm", [answer[:40]])
    return Classification(GENERAL, 0.0, "default")
