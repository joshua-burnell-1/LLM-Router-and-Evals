"""Eval-gated routing: pick the model with the best measured trade-off for the query.

    score = w_quality * quality - w_cost * norm_cost - w_latency * norm_latency

Quality, cost and latency are per-category means from the leaderboard; cost and latency are
normalized by the most expensive / slowest candidate. A mode may also refuse models whose
quality is more than `max_quality_drop` below the best, or below an absolute `quality_floor`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from routewise.config import config_path, default_db_path, load_yaml
from routewise.evals.store import ResultStore
from routewise.models.client import ChatResult, Message, chat
from routewise.models.registry import ModelSpec, flagship, load_models
from routewise.router.classifier import (
    GENERAL,
    Classification,
    ExampleIndex,
    classify,
    classify_async,
)
from routewise.router.leaderboard import OVERALL, CellStats, Leaderboard


class NoDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModeWeights:
    quality: float
    cost: float
    latency: float
    max_quality_drop: float | None = None
    quality_floor: float | None = None


@dataclass
class RouterConfig:
    modes: dict[str, ModeWeights]
    min_samples: int = 1
    default_mode: str = "balanced"
    llm_fallback: bool = False
    llm_model: str = "claude-haiku-5-5"


def load_router_config(path: str | Path | None = None) -> RouterConfig:
    data = load_yaml(path or config_path("router.yaml"))
    classifier = data.get("classifier", {})
    return RouterConfig(
        modes={name: ModeWeights(**w) for name, w in data["modes"].items()},
        min_samples=int(data.get("min_samples", 1)),
        default_mode=data.get("default_mode", "balanced"),
        llm_fallback=bool(classifier.get("llm_fallback", False)),
        llm_model=classifier.get("llm_model", "claude-haiku-5-5"),
    )


@dataclass
class Candidate:
    model_id: str
    score: float
    quality: float
    mean_cost_usd: float
    mean_latency_s: float
    norm_cost: float
    norm_latency: float
    n: int
    category: str
    eligible: bool = True
    excluded_reason: str = ""


@dataclass
class RouteDecision:
    model_id: str
    mode: str
    category: str
    used_category: str
    classification: Classification
    candidates: list[Candidate] = field(default_factory=list)
    rationale: str = ""


def _why(winner: Candidate, runner: Candidate) -> str:
    """One clause explaining why `winner` outranked `runner`, from the actual numbers."""
    dq = winner.quality - runner.quality
    if runner.mean_cost_usd > 0:
        ratio = winner.mean_cost_usd / runner.mean_cost_usd
    else:
        ratio = 1.0 if winner.mean_cost_usd == 0 else float("inf")
    cost = f"{ratio:.0%} of the cost" if ratio <= 1 else f"{ratio:.1f}x the cost"
    if dq > 0 and ratio <= 1:
        return f"it scores {dq:+.2f} quality at {cost}"
    if dq == 0 and ratio < 1:
        return f"it matches quality at {cost}"
    if dq < 0 and ratio < 1:
        return f"it gives up only {-dq:.2f} quality for {cost}"
    if dq > 0:
        return f"its {dq:+.2f} quality gain is worth {cost} in this mode"
    faster = (
        f"it is faster ({winner.mean_latency_s:.1f}s vs {runner.mean_latency_s:.1f}s)"
        if winner.mean_latency_s < runner.mean_latency_s
        else "it scores higher on the mode's weights"
    )
    return faster + (f" at {cost}" if ratio != 1 else "")


class Router:
    def __init__(
        self,
        leaderboard: Leaderboard,
        config: RouterConfig | None = None,
        registry: Mapping[str, ModelSpec] | None = None,
        models: list[str] | None = None,
        examples: ExampleIndex | None = None,
    ):
        self.examples = examples
        self.leaderboard = leaderboard
        self.config = config or load_router_config()
        self.registry = registry if registry is not None else load_models(include_disabled=True)
        self.models = models or leaderboard.models()

    def _cell(self, model: str, category: str) -> CellStats | None:
        cell = self.leaderboard.get(model, category)
        if cell is None or cell.n < self.config.min_samples:
            return None
        return cell

    def rank(self, category: str, mode: str) -> tuple[list[Candidate], str]:
        """Candidates best-first, and the category whose stats were used."""
        if mode not in self.config.modes:
            raise ValueError(f"unknown mode {mode!r}; choose from {sorted(self.config.modes)}")
        weights = self.config.modes[mode]
        used = category
        cells = {m: self._cell(m, category) for m in self.models} if category != GENERAL else {}
        cells = {m: c for m, c in cells.items() if c is not None}
        if len(cells) < 2:
            used = OVERALL
            cells = {m: c for m in self.models if (c := self._cell(m, OVERALL)) is not None}
        if not cells:
            raise NoDataError("no eval results yet; run the eval suite first")

        max_cost = max(c.mean_cost_usd for c in cells.values()) or 1.0
        max_latency = max(c.mean_latency_s for c in cells.values()) or 1.0
        best_quality = max(c.mean_quality for c in cells.values())
        candidates = []
        for model, cell in cells.items():
            norm_cost = cell.mean_cost_usd / max_cost
            norm_latency = cell.mean_latency_s / max_latency
            score = (
                weights.quality * cell.mean_quality
                - weights.cost * norm_cost
                - weights.latency * norm_latency
            )
            cand = Candidate(
                model,
                score,
                cell.mean_quality,
                cell.mean_cost_usd,
                cell.mean_latency_s,
                norm_cost,
                norm_latency,
                cell.n,
                used,
            )
            if weights.max_quality_drop is not None and (
                cell.mean_quality < best_quality - weights.max_quality_drop
            ):
                cand.eligible = False
                cand.excluded_reason = f"more than {weights.max_quality_drop} below best quality"
            if weights.quality_floor is not None and cell.mean_quality < weights.quality_floor:
                cand.eligible = False
                cand.excluded_reason = f"below quality floor {weights.quality_floor}"
            candidates.append(cand)
        candidates.sort(key=lambda c: (c.eligible, c.score, c.quality), reverse=True)
        return candidates, used

    def _decide(
        self, classification: Classification, mode: str, category: str | None
    ) -> RouteDecision:
        category = category or classification.category
        candidates, used = self.rank(category, mode)
        winner = candidates[0]
        return RouteDecision(
            model_id=winner.model_id,
            mode=mode,
            category=category,
            used_category=used,
            classification=classification,
            candidates=candidates,
            rationale=self._rationale(category, used, mode, candidates, classification),
        )

    def route(
        self, query: str, mode: str | None = None, *, category: str | None = None
    ) -> RouteDecision:
        classification = classify(query, self.examples)
        return self._decide(classification, mode or self.config.default_mode, category)

    async def aroute(
        self,
        query: str,
        mode: str | None = None,
        *,
        category: str | None = None,
        chat_fn: Callable[..., Awaitable[ChatResult]] = chat,
    ) -> RouteDecision:
        classification = await classify_async(
            query,
            use_llm=self.config.llm_fallback,
            llm_model=self.config.llm_model,
            chat_fn=chat_fn,
            index=self.examples,
        )
        return self._decide(classification, mode or self.config.default_mode, category)

    def _rationale(
        self,
        category: str,
        used: str,
        mode: str,
        candidates: list[Candidate],
        classification: Classification,
    ) -> str:
        shown = ", ".join(classification.matched[:3])
        how = {
            "knn": f"nearest eval examples: {shown}",
            "rules": "rules matched " + ", ".join(repr(m) for m in classification.matched[:3]),
            "llm": "classified by an LLM",
        }.get(classification.method, "no rule or example matched")
        parts = [f"Category: {category} ({how})."]
        if category == GENERAL:
            parts.append("No category matched, so overall scores were used.")
        elif used != category:
            parts.append("Too little per-category data, so overall scores were used.")
        winner = candidates[0]
        parts.append(
            f"Mode: {mode}. Winner {winner.model_id} (score {winner.score:.2f}: quality "
            f"{winner.quality:.2f}, ${winner.mean_cost_usd:.5f}/query, "
            f"{winner.mean_latency_s:.1f}s)"
        )
        if len(candidates) > 1:
            runner = candidates[1]
            parts.append(
                f"over {runner.model_id} (score {runner.score:.2f}: quality {runner.quality:.2f}, "
                f"${runner.mean_cost_usd:.5f}/query, {runner.mean_latency_s:.1f}s)"
                + (
                    f", which was excluded: {runner.excluded_reason}."
                    if not runner.eligible
                    else f": {_why(winner, runner)}."
                )
            )
        else:
            parts.append("as the only model with data.")
        return " ".join(parts)

    def flagship_model(self) -> str:
        specs = [self.registry[m] for m in self.models if m in self.registry]
        if not specs:
            raise NoDataError("no models with eval results")
        return flagship(specs).id

    def always_flagship(self, query: str) -> RouteDecision:
        """Baseline for savings: always send the query to the most expensive model."""
        classification = classify(query, self.examples)
        model = self.flagship_model()
        candidates, used = self.rank(classification.category, "quality")
        chosen = [c for c in candidates if c.model_id == model]
        if not chosen:  # flagship lacks data in this category: describe it by overall stats
            overall, _ = self.rank(OVERALL, "quality")
            chosen = [c for c in overall if c.model_id == model]
        return RouteDecision(
            model_id=model,
            mode="flagship",
            category=classification.category,
            used_category=used,
            classification=classification,
            candidates=chosen + [c for c in candidates if c.model_id != model],
            rationale=f"Baseline: always the flagship (most expensive) model, {model}.",
        )

    async def complete(
        self,
        query: str | list[Message],
        mode: str | None = None,
        *,
        chat_fn: Callable[..., Awaitable[ChatResult]] = chat,
        max_tokens: int = 1024,
        **chat_kwargs: Any,
    ) -> tuple[RouteDecision, ChatResult, list[str]]:
        """Route, call the winner, and fail over down the ranking if a call errors."""
        messages = [{"role": "user", "content": query}] if isinstance(query, str) else query
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = f"{system}\n\n{last_user}" if system else last_user
        decision = await self.aroute(text, mode, chat_fn=chat_fn)
        attempts: list[str] = []
        last_error: Exception | None = None
        for cand in decision.candidates:
            attempts.append(cand.model_id)
            try:
                result = await chat_fn(
                    cand.model_id, messages, max_tokens=max_tokens, **chat_kwargs
                )
                return decision, result, attempts
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"all candidates failed: {attempts}") from last_error


_DEFAULT_ROUTER: Router | None = None


def default_router(refresh: bool = False) -> Router:
    """A Router over data/results.db (or $ROUTEWISE_DB) and the repo configs."""
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None or refresh:
        from routewise.evals.loader import all_tasks

        tasks = all_tasks()
        _DEFAULT_ROUTER = Router(
            Leaderboard.from_store(ResultStore(default_db_path()), current=tasks),
            examples=ExampleIndex.from_tasks(tasks),
        )
    return _DEFAULT_ROUTER


def route(query: str, mode: str = "balanced") -> RouteDecision:
    return default_router().route(query, mode)


def always_flagship(query: str) -> RouteDecision:
    return default_router().always_flagship(query)
