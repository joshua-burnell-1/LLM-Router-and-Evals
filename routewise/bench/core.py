"""Blind benchmarks: run the same tasks across models, rate responses without knowing which
model wrote them, then reveal which model you preferred.

A benchmark is a fixed, versioned prompt (a realistic task from your own work) plus guidance
for the rater. Because prompts stay fixed, sessions run months apart remain comparable: when a
new model ships, run it against your current model on the same benchmarks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import string
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from routewise.bench.store import BenchStore, Output
from routewise.config import DATA_DIR, load_yaml
from routewise.models.client import ChatResult, chat

BUILTIN_DIR = DATA_DIR / "benchmarks"
CUSTOM_DIR = BUILTIN_DIR / "custom"

DEFAULT_SYSTEM_PROMPT = (
    "You are an experienced senior product manager. Answer the request directly and concretely, "
    "using only the facts given plus clearly labelled assumptions."
)
# Appended to every benchmark prompt so responses do not give away which model wrote them.
IDENTITY_RULE = "Do not mention your name, which AI model you are, or the company that made you."

_IDENTITY_RE = re.compile(
    r"\b(?:claude(?:[- ][\w.-]+)?|anthropic|chatgpt|gpt[-\w.]*|openai|gemini(?:[- ][\w.-]+)?|"
    r"bard|llama(?:[- ][\w.-]+)?|mistral|deepseek|grok)\b",
    re.I,
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def redact_identity(text: str) -> str:
    """Mask model and vendor names so a response cannot unblind the rater."""
    return _IDENTITY_RE.sub("[model]", text)


@dataclass(frozen=True)
class Benchmark:
    id: str
    title: str
    prompt: str
    suite: str = "custom"
    guidance: str = ""
    system_prompt: str | None = None
    tags: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        """Hash of what the model sees. Edit the prompt and old results stop comparing."""
        blob = json.dumps([self.prompt, self.system_prompt])
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def messages(self) -> list[dict[str, str]]:
        system = f"{self.system_prompt or DEFAULT_SYSTEM_PROMPT}\n\n{IDENTITY_RULE}"
        return [{"role": "system", "content": system}, {"role": "user", "content": self.prompt}]


def _from_file(path: Path) -> list[Benchmark]:
    data = load_yaml(path)
    suite = data.get("suite", "custom")
    out = []
    for raw in data.get("benchmarks", []):
        out.append(
            Benchmark(
                id=str(raw["id"]),
                title=raw["title"],
                prompt=raw["prompt"].strip(),
                suite=raw.get("suite", suite),
                guidance=(raw.get("guidance") or "").strip(),
                system_prompt=raw.get("system_prompt"),
                tags=tuple(raw.get("tags", ())),
            )
        )
    return out


def load_benchmarks(
    builtin_dir: str | Path = BUILTIN_DIR, custom_dir: str | Path | None = CUSTOM_DIR
) -> dict[str, Benchmark]:
    """Built-in benchmarks plus your own (custom ids may not clash with built-in ones)."""
    paths = sorted(Path(builtin_dir).glob("*.y*ml")) if Path(builtin_dir).exists() else []
    if custom_dir is not None and Path(custom_dir).exists():
        paths += sorted(Path(custom_dir).glob("*.y*ml"))
    benches: dict[str, Benchmark] = {}
    for path in paths:
        for b in _from_file(path):
            if b.id in benches:
                raise ValueError(f"duplicate benchmark id {b.id!r} in {path}")
            benches[b.id] = b
    return benches


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "benchmark"


def save_custom_benchmark(bench: Benchmark, custom_dir: str | Path = CUSTOM_DIR) -> Path:
    """Write a user-created benchmark to its own YAML file and return the path."""
    if not bench.title.strip() or not bench.prompt.strip():
        raise ValueError("a benchmark needs a title and a prompt")
    bench_id = bench.id or slugify(bench.title)
    custom_dir = Path(custom_dir)
    custom_dir.mkdir(parents=True, exist_ok=True)
    path = custom_dir / f"{bench_id}.yaml"
    taken = set(load_benchmarks(BUILTIN_DIR, custom_dir))
    if path.exists() or bench_id in taken:
        raise ValueError(f"a benchmark with id {bench_id!r} already exists")
    entry = {"id": bench_id, "title": bench.title.strip(), "prompt": bench.prompt.strip()}
    if bench.guidance.strip():
        entry["guidance"] = bench.guidance.strip()
    if bench.system_prompt:
        entry["system_prompt"] = bench.system_prompt
    if bench.tags:
        entry["tags"] = list(bench.tags)
    doc = {"suite": bench.suite or "custom", "benchmarks": [entry]}
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100))
    return path


# --- sessions -------------------------------------------------------------------------------


def create_session(
    store: BenchStore,
    benchmarks: Iterable[Benchmark],
    models: list[str],
    *,
    baseline: str,
    name: str | None = None,
    seed: int | None = None,
) -> str:
    """Create a session with every (benchmark, model) pair, each benchmark's responses given
    labels A, B, C... in an independently shuffled order."""
    benchmarks = list(benchmarks)
    models = list(dict.fromkeys(models))
    if len(models) < 2:
        raise ValueError("pick at least two models to compare")
    if baseline not in models:
        raise ValueError("the baseline (your current model) must be one of the models")
    if not benchmarks:
        raise ValueError("pick at least one benchmark")
    if len(models) > len(string.ascii_uppercase):
        raise ValueError("too many models for one session")
    seed = random.SystemRandom().randrange(2**31) if seed is None else seed
    from routewise.bench.store import Session

    session = Session(
        session_id=datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        name=name,
        created_at=now_iso(),
        baseline_model=baseline,
        models=models,
        benchmark_ids=[b.id for b in benchmarks],
        seed=seed,
        revealed_at=None,
    )
    rows = []
    for bench in benchmarks:
        order = models[:]
        random.Random(f"{seed}:{bench.id}").shuffle(order)
        for label, model in zip(string.ascii_uppercase, order, strict=False):
            rows.append((bench.id, bench.version, model, label))
    store.add_session(session, rows)
    return session.session_id


async def generate_outputs(
    store: BenchStore,
    session_id: str,
    benchmarks: Mapping[str, Benchmark],
    *,
    chat_fn: Callable[..., Awaitable[ChatResult]] = chat,
    concurrency: int = 4,
    max_tokens: int = 2000,
    progress: Callable[[str], None] | None = None,
) -> int:
    """Generate every missing response (errored ones are retried). Returns how many succeeded."""
    pending = [o for o in store.outputs(session_id) if o.output is None]
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def run(o: Output) -> None:
        nonlocal done
        bench = benchmarks[o.benchmark_id]
        async with semaphore:
            try:
                result = await chat_fn(o.model_id, bench.messages(), max_tokens=max_tokens)
            except Exception as exc:
                store.set_output(
                    o.id,
                    output=None,
                    error=f"{type(exc).__name__}: {exc}"[:1000],
                    latency_s=None,
                    cost_usd=None,
                    created_at=now_iso(),
                )
                if progress:
                    progress(f"{bench.title} / response {o.label}: error")
                return
        store.set_output(
            o.id,
            output=result.text,
            error=None,
            latency_s=result.latency_s,
            cost_usd=result.cost_usd,
            created_at=now_iso(),
        )
        done += 1
        if progress:
            progress(f"{bench.title} / response {o.label}: ready")

    await asyncio.gather(*(run(o) for o in pending))
    return done


def rate(
    store: BenchStore, output_id: int, score: int, liked: str = "", disliked: str = ""
) -> None:
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("score must be a whole number from 1 to 5")
    liked, disliked = (liked or "").strip(), (disliked or "").strip()
    if not liked and not disliked:
        raise ValueError("add a sentence on what you liked or disliked")
    if store.output(output_id).output is None:
        raise ValueError("this response has no output yet")
    store.set_rating(output_id, score, liked, disliked, now_iso())


def session_progress(store: BenchStore, session_id: str) -> tuple[int, int]:
    """(rated, total) responses."""
    outputs = store.outputs(session_id)
    return sum(o.score is not None for o in outputs), len(outputs)


class NotCompleteError(RuntimeError):
    pass


@dataclass
class ModelSummary:
    model_id: str
    mean_score: float
    n: int
    wins: float
    delta_vs_baseline: float
    is_baseline: bool
    liked: list[str] = field(default_factory=list)
    disliked: list[str] = field(default_factory=list)


@dataclass
class RevealRow:
    benchmark_id: str
    label: str
    model_id: str
    score: int
    liked: str
    disliked: str


@dataclass
class Reveal:
    session_id: str
    baseline: str
    preferred: str
    models: list[ModelSummary]
    rows: list[RevealRow]


def reveal(store: BenchStore, session_id: str) -> Reveal:
    """Unblind a fully rated session."""
    session = store.session(session_id)
    outputs = store.outputs(session_id)
    rated, total = session_progress(store, session_id)
    if rated < total:
        raise NotCompleteError(f"{total - rated} of {total} responses still need a rating")

    scores: dict[str, list[int]] = defaultdict(list)
    wins: dict[str, float] = defaultdict(float)
    by_bench: dict[str, list[Output]] = defaultdict(list)
    for o in outputs:
        scores[o.model_id].append(o.score)
        by_bench[o.benchmark_id].append(o)
    for group in by_bench.values():
        top = max(o.score for o in group)
        winners = [o for o in group if o.score == top]
        for o in winners:
            wins[o.model_id] += 1 / len(winners)

    means = {m: sum(s) / len(s) for m, s in scores.items()}
    base = means[session.baseline_model]
    summaries = [
        ModelSummary(
            model_id=m,
            mean_score=means[m],
            n=len(scores[m]),
            wins=wins[m],
            delta_vs_baseline=means[m] - base,
            is_baseline=m == session.baseline_model,
            liked=[o.liked for o in outputs if o.model_id == m and o.liked],
            disliked=[o.disliked for o in outputs if o.model_id == m and o.disliked],
        )
        for m in session.models
    ]
    summaries.sort(key=lambda s: (-s.mean_score, -s.wins, s.model_id))
    store.mark_revealed(session_id, now_iso())
    rows = [
        RevealRow(o.benchmark_id, o.label, o.model_id, o.score, o.liked or "", o.disliked or "")
        for o in outputs
    ]
    return Reveal(session_id, session.baseline_model, summaries[0].model_id, summaries, rows)


@dataclass
class HistoryRow:
    model_id: str
    mean_score: float
    ratings: int
    sessions: int
    first_at: str
    last_at: str


def history(store: BenchStore, benchmarks: Mapping[str, Benchmark]) -> list[HistoryRow]:
    """Human scores per model across all revealed sessions, counting only ratings made on the
    current version of each benchmark so old and new results stay comparable."""
    scores: dict[str, list[int]] = defaultdict(list)
    sessions: dict[str, set[str]] = defaultdict(set)
    dates: dict[str, list[str]] = defaultdict(list)
    for session in store.sessions():
        if not session.revealed_at:
            continue
        for o in store.outputs(session.session_id):
            bench = benchmarks.get(o.benchmark_id)
            if bench is None or bench.version != o.benchmark_version or o.score is None:
                continue
            scores[o.model_id].append(o.score)
            sessions[o.model_id].add(session.session_id)
            dates[o.model_id].append(session.created_at)
    rows = [
        HistoryRow(m, sum(s) / len(s), len(s), len(sessions[m]), min(dates[m]), max(dates[m]))
        for m, s in scores.items()
    ]
    return sorted(rows, key=lambda r: -r.mean_score)
