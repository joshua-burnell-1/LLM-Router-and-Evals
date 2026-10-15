"""Estimate, run and summarize a studio benchmark."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from routewise.evals.framework import LLMJudge
from routewise.evals.rescore import rescore
from routewise.evals.runner import RunSummary, run_eval
from routewise.evals.store import ResultStore
from routewise.models.client import ChatResult, estimate_tokens
from routewise.models.registry import ModelSpec, cost_usd
from routewise.studio.catalog import CATALOG, build_scorer_spec
from routewise.studio.model import StudioBenchmark

ASSUMED_OUTPUT_TOKENS = 600  # answers plus some reasoning; deliberately on the high side
JUDGE_OUTPUT_TOKENS = 120


@dataclass
class CostEstimate:
    calls: int
    total_usd: float
    per_model: dict[str, float]
    judge_usd: float
    assumed_output_tokens: int
    cached: int = 0  # answers already stored; rerunning them is free


def estimate_cost(
    bench: StudioBenchmark,
    models: Sequence[str],
    registry: Mapping[str, ModelSpec],
    *,
    judge_model: str | None = None,
    output_tokens: int = ASSUMED_OUTPUT_TOKENS,
    store: ResultStore | None = None,
) -> CostEstimate:
    """Rough upper-end cost of the new answers a run needs (list prices). With a store,
    answers already recorded for the current prompt version are free and counted as cached."""
    tasks = bench.to_tasks() if not bench.validate() else []
    per_model: dict[str, float] = {}
    judge_usd = 0.0
    judge_spec = registry.get(judge_model) if judge_model else None
    calls = cached = 0
    for model in models:
        spec = registry[model]
        total = 0.0
        for task in tasks:
            if store is not None and store.has(task.id, task.version, model):
                cached += 1
                continue
            calls += 1
            prompt_tokens = estimate_tokens((task.system_prompt or "") + task.prompt)
            total += cost_usd(spec, prompt_tokens, output_tokens)
            if bench.uses_judge() and judge_spec is not None:
                judge_in = prompt_tokens + output_tokens + 250
                judge_usd += cost_usd(judge_spec, judge_in, JUDGE_OUTPUT_TOKENS)
        per_model[model] = total
    return CostEstimate(
        calls=calls,
        total_usd=sum(per_model.values()) + judge_usd,
        per_model=per_model,
        judge_usd=judge_usd,
        assumed_output_tokens=output_tokens,
        cached=cached,
    )


async def run_studio(
    bench: StudioBenchmark,
    models: Sequence[str],
    store: ResultStore,
    *,
    chat_fn: Callable[..., Awaitable[ChatResult]],
    judge: LLMJudge | None,
    progress: Callable[[str], None] = print,
    max_cost: float | None = None,
    concurrency: int = 4,
    force: bool = False,
) -> RunSummary:
    """Run every test case on every model (skipping ones already answered), then rescore all
    stored answers with the current checks so edited checks apply without new model calls."""
    tasks = bench.to_tasks()
    summary = await run_eval(
        tasks,
        list(models),
        store,
        judge=judge,
        chat_fn=chat_fn,
        progress=progress,
        max_cost=max_cost,
        concurrency=concurrency,
        force=force,
        run_args={"studio": bench.slug, "models": list(models)},
    )
    await rescore(store, tasks, judge=None)
    return summary


@dataclass
class CheckScore:
    label: str
    value: float
    reason: str


@dataclass
class CaseResult:
    case_index: int
    model_id: str
    score: float | None
    output: str | None
    checks: list[CheckScore] = field(default_factory=list)
    cost_usd: float | None = None
    latency_s: float | None = None
    error: str | None = None


@dataclass
class ModelResult:
    model_id: str
    cases: int
    mean_score: float
    pass_rate: float
    mean_cost_usd: float
    total_cost_usd: float
    mean_latency_s: float
    checks: dict[str, float]
    errors: int = 0


@dataclass
class StudioResults:
    models: list[ModelResult]
    cases: list[CaseResult]
    best_quality: str | None
    best_value: str | None
    pass_threshold: float
    total_cases: int

    def tied_best(self, tolerance: float = 0.005) -> list[str]:
        """Models whose mean score is level with the top score (best first)."""
        if not self.models:
            return []
        top = self.models[0].mean_score
        return [m.model_id for m in self.models if m.mean_score >= top - tolerance]


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def verdict(results: StudioResults, name: Callable[[str], str]) -> tuple[str, str]:
    """(headline, one-line explanation) in plain language; calls out ties honestly."""
    if not results.models:
        return "No results yet", "Run the benchmark to compare models."
    by_id = {m.model_id: m for m in results.models}
    best = results.models[0]
    value = by_id[results.best_value or best.model_id]
    tied = results.tied_best()
    if len(tied) > 1:
        priciest = max((by_id[m] for m in tied), key=lambda m: m.mean_cost_usd)
        share = value.mean_cost_usd / priciest.mean_cost_usd if priciest.mean_cost_usd else 1.0
        head = f"{len(tied)} models tie at {best.mean_score:.2f}."
        if value.model_id != priciest.model_id:
            head += f" {name(value.model_id)} does it for {share:.0%} of the cost."
        detail = (
            f"{_join([name(m) for m in tied])} all averaged {best.mean_score:.2f} on your checks"
        )
        if value.model_id != priciest.model_id:
            detail += f", so paying for {name(priciest.model_id)} buys nothing here."
        else:
            detail += "."
        if value.model_id not in tied:
            detail += f" {name(value.model_id)} averaged {value.mean_score:.2f} for even less."
        return head, detail
    if value.model_id == best.model_id:
        return (
            f"{name(best.model_id)} wins on quality and value",
            f"It averaged {best.mean_score:.2f} and passed {best.pass_rate:.0%} of test cases.",
        )
    share = value.mean_cost_usd / best.mean_cost_usd if best.mean_cost_usd else 0
    return (
        f"Best quality: {name(best.model_id)}. Best value: {name(value.model_id)}.",
        f"{name(best.model_id)} averaged {best.mean_score:.2f}. {name(value.model_id)} averaged "
        f"{value.mean_score:.2f} at {share:.0%} of the cost, within 0.05 of the best.",
    )


def _check_labels(bench: StudioBenchmark) -> list[tuple[str, str]]:
    """(scorer type, human label) for each configured check, in scoring order."""
    out = []
    for check in bench.checks:
        try:
            spec = build_scorer_spec(check.key, check.values, check.weight, check.required)
        except ValueError:
            continue
        out.append((spec.type, CATALOG[check.key].label))
    return out


def _label_components(detail: list[dict], labels: list[tuple[str, str]]) -> list[CheckScore]:
    """Match stored score components (scorer types) to the configured checks, in order."""
    out, i = [], 0
    for comp in detail:
        while i < len(labels) and labels[i][0] != comp["scorer"]:
            i += 1
        label = labels[i][1] if i < len(labels) else comp["scorer"]
        out.append(CheckScore(label, comp["value"], comp["reason"]))
        i += 1
    return out


def summarize(
    bench: StudioBenchmark,
    models: Sequence[str],
    store: ResultStore,
    registry: Mapping[str, ModelSpec] | None = None,
    *,
    pass_threshold: float = 0.8,
    value_margin: float = 0.05,
) -> StudioResults:
    """Per-model and per-case results for the benchmark's current prompt version."""
    tasks = bench.to_tasks()
    index = {t.id: i for i, t in enumerate(tasks)}
    all_rows = store.rows(models=list(models), include_errors=True, current=tasks)
    labels = _check_labels(bench)

    cases: list[CaseResult] = []
    per_model: dict[str, list] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    for row in all_rows:
        if row.error:
            errors[row.model_id] += 1
            cases.append(
                CaseResult(index[row.task_id], row.model_id, None, row.output, error=row.error)
            )
            continue
        detail = json.loads(row.score_detail or "[]")
        case = CaseResult(
            index[row.task_id],
            row.model_id,
            row.quality,
            row.output,
            _label_components(detail, labels),
            row.cost_usd,
            row.latency_s,
        )
        cases.append(case)
        per_model[row.model_id].append(case)

    results = []
    for model in models:
        rows = per_model.get(model, [])
        if not rows:
            continue
        check_values: dict[str, list[float]] = defaultdict(list)
        for c in rows:
            for chk in c.checks:
                check_values[chk.label].append(chk.value)
        n = len(rows)
        results.append(
            ModelResult(
                model_id=model,
                cases=n,
                mean_score=sum(c.score for c in rows) / n,
                pass_rate=sum(c.score >= pass_threshold for c in rows) / n,
                mean_cost_usd=sum(c.cost_usd or 0 for c in rows) / n,
                total_cost_usd=sum(c.cost_usd or 0 for c in rows),
                mean_latency_s=sum(c.latency_s or 0 for c in rows) / n,
                checks={k: sum(v) / len(v) for k, v in check_values.items()},
                errors=errors.get(model, 0),
            )
        )
    results.sort(key=lambda r: (-r.mean_score, r.mean_cost_usd, r.model_id))
    best_quality = results[0].model_id if results else None
    best_value = None
    if results:
        bar = results[0].mean_score - value_margin
        good_enough = [r for r in results if r.mean_score >= bar]
        best_value = min(good_enough, key=lambda r: (r.mean_cost_usd, -r.mean_score)).model_id
    cases.sort(key=lambda c: (c.case_index, c.model_id))
    return StudioResults(results, cases, best_quality, best_value, pass_threshold, len(tasks))
