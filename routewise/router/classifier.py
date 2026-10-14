"""Classify a query into one of the eval categories.

Three layers, cheapest signal first:
1. Nearest neighbours over labelled eval prompts (TF-IDF cosine). The eval set doubles as
   routing training data, so every task you add also teaches the classifier.
2. Keyword rules (small and readable). Matches inside the instruction (the first 200
   characters) count double, so pasted documents do not drown out the actual request.
3. Optionally, a cheap LLM for queries nothing else places.
Queries nothing matches are "general" and are routed on overall scores.

Classify the whole request the model will see (system prompt plus user turn): the system
prompt often defines the task.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

from routewise.evals.framework import CATEGORIES, EvalTask
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
    ("structured_output", r"\bexactly (?:\d+|one|two|three|four|five)\b", 1),
    ("structured_output", r"\bclassify\b|\bcategori[sz]e\b|\btriage\b", 1),
    ("reasoning", r"\bhow many\b|\bhow much\b", 1),
    ("reasoning", r"\bcalculate\b|\bcompute\b|\bsolve\b", 2),
    ("reasoning", r"\bpuzzle\b|\bconstraints?\b|\bschedule\b", 1),
    ("reasoning", r"\bbug\b|\bdebug\b|line number", 2),
    ("reasoning", r"\bflaw\b|\bweakness\b|\bwhat'?s wrong\b", 2),
    ("reasoning", r"\bstep by step\b|\bprove\b|\bthink (?:it )?through\b", 1),
    ("reasoning", r"\bbrief working\b|\bshow (?:your )?work", 1),
    ("reasoning", r"\bwhat is the total\b|\btotal cost\b|\bwhat weekday\b", 1),
    ("summarization", r"\bsummar", 2),
    ("summarization", r"\btl;?dr\b|\brecap\b|\bkey points\b", 2),
    ("summarization", r"\baction items?\b|\bmeeting notes\b", 2),
    ("summarization", r"\bcondense\b|\bshorten\b", 1),
    ("voice", r"\btranscript\b|\basr\b|\bspeech\b", 2),
    ("voice", r"\buh\b|\bum+\b", 1),
    ("voice", r"barge[- ]in|\binterrupt", 2),
    ("voice", r"\bspoken\b|\bvoice\b|\bphone\b|\bcaller\b", 1),
    ("voice", r"\bangry\b|\bupset\b|\bfrustrated\b", 1),
    ("voice", r"\bintent\b|\bslots?\b", 1),
    ("safety", r"ignore (all )?(previous|prior|above) instructions", 3),
    ("safety", r"\bsystem prompt\b|\bjailbreak\b", 2),
    ("safety", r"\bnever reveal\b|\bdo not reveal\b|\bnever follow instructions\b", 3),
    ("safety", r"\bredact\b|\bpii\b|\bpersonal data\b", 2),
    ("safety", r"[\w.+-]+@[\w-]+\.\w+", 1),
    ("safety", r"\bpassword\b|\bsecret\b|\bapi key\b|\bconfidential\b", 1),
]

_COMPILED = [(cat, re.compile(pattern, re.I), weight) for cat, pattern, weight in RULES]
INSTRUCTION_WINDOW = 200  # characters; rule matches inside it count double


@dataclass
class Classification:
    category: str
    confidence: float
    method: str  # rules | llm | default
    matched: list[str] = field(default_factory=list)


_TOKEN = re.compile(r"[a-z][a-z0-9']+")
_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "your",
        "you",
        "are",
        "was",
        "were",
        "what",
        "which",
        "into",
        "have",
        "has",
        "had",
        "not",
        "but",
        "can",
        "will",
        "would",
        "should",
        "could",
        "about",
        "then",
        "than",
        "them",
        "they",
        "their",
        "there",
        "here",
        "when",
        "where",
        "who",
        "how",
        "its",
        "it's",
        "our",
        "out",
        "all",
        "any",
        "each",
        "only",
        "one",
        "two",
        "its",
        "also",
        "just",
        "like",
        "more",
        "most",
        "such",
        "very",
        "being",
        "been",
        "does",
        "did",
        "done",
        "some",
        "same",
        "other",
        "over",
        "under",
        "again",
    ]
)


def task_text(task: EvalTask) -> str:
    """The text a router sees for a task: system prompt plus user prompt."""
    return f"{task.system_prompt}\n\n{task.prompt}" if task.system_prompt else task.prompt


def _tokens(text: str) -> list[str]:
    words = (w.rstrip("s") if len(w) > 4 else w for w in _TOKEN.findall(text.lower()))
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


class ExampleIndex:
    """TF-IDF nearest-neighbour classifier over labelled examples (id, text, category)."""

    def __init__(
        self, examples: Iterable[tuple[str, str, str]], *, k: int = 3, threshold: float = 0.3
    ):
        self.examples = list(examples)
        self.k = k
        self.threshold = threshold
        docs = [Counter(_tokens(text)) for _, text, _ in self.examples]
        df: Counter = Counter()
        for doc in docs:
            df.update(doc.keys())
        n = len(docs)
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self._vectors = [self._vector(doc) for doc in docs]

    @classmethod
    def from_tasks(cls, tasks: Iterable[EvalTask], **kwargs: float) -> ExampleIndex:
        return cls(((t.id, task_text(t), t.category) for t in tasks), **kwargs)

    def without(self, example_id: str) -> ExampleIndex:
        kept = [e for e in self.examples if e[0] != example_id]
        return ExampleIndex(kept, k=self.k, threshold=self.threshold)

    def _vector(self, counts: Counter) -> dict[str, float]:
        vec = {t: c * self._idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def nearest(self, text: str, k: int | None = None) -> list[tuple[float, str, str]]:
        query = self._vector(Counter(_tokens(text)))
        sims = []
        for (ex_id, _, category), vec in zip(self.examples, self._vectors, strict=True):
            sim = sum(w * vec.get(t, 0.0) for t, w in query.items())
            sims.append((sim, ex_id, category))
        sims.sort(reverse=True)
        return sims[: k or self.k]

    def predict(self, text: str) -> Classification | None:
        hits = [h for h in self.nearest(text) if h[0] >= self.threshold]
        if not hits:
            return None
        votes: Counter = Counter()
        for sim, _, category in hits:
            votes[category] += sim
        best = max(votes, key=lambda c: (votes[c], -CATEGORIES.index(c)))
        return Classification(
            best,
            votes[best] / sum(votes.values()),
            "knn",
            [f"{ex_id} ({sim:.2f})" for sim, ex_id, cat in hits if cat == best],
        )


def classify(query: str, index: ExampleIndex | None = None) -> Classification:
    if index is not None:
        knn = index.predict(query)
        if knn is not None:
            return knn
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for category, regex, weight in _COMPILED:
        m = regex.search(query)
        if m:
            boost = 2 if m.start() < INSTRUCTION_WINDOW else 1
            scores[category] = scores.get(category, 0.0) + weight * boost
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
    index: ExampleIndex | None = None,
) -> Classification:
    """Neighbours and rules first; when both miss and `use_llm` is set, ask a cheap model."""
    result = classify(query, index)
    if result.method != "default" or not use_llm:
        return result
    reply = await chat_fn(
        llm_model, [{"role": "user", "content": LLM_PROMPT + query}], max_tokens=20
    )
    answer = reply.text.strip().lower()
    for category in CATEGORIES:
        if category in answer:
            return Classification(category, 0.5, "llm", [answer[:40]])
    return Classification(GENERAL, 0.0, "default")
