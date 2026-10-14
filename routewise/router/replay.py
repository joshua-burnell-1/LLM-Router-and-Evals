"""Offline router evaluation: replay the eval set through the router.

For each task, the router decides using a leaderboard built WITHOUT that task's results
(leave-one-out), so a task never vouches for itself. Outcomes come from the recorded results
of whichever model was picked. Compared policies: the router, always-flagship, every fixed
model, and an oracle (cheapest model that achieved the task's best quality: an upper bound).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from routewise.evals.framework import EvalTask
from routewise.evals.store import ResultRow
from routewise.models.registry import ModelSpec, flagship
from routewise.router.classifier import classify
from routewise.router.leaderboard import Leaderboard
from routewise.router.router import NoDataError, Router, RouterConfig


@dataclass
class PolicyOutcome:
    name: str
    n_tasks: int = 0
    quality_sum: float = 0.0
    total_cost_usd: float = 0.0
    latency_sum: float = 0.0
    picks: Counter = field(default_factory=Counter)

    def add(self, model: str, quality: float, cost: float, latency: float) -> None:
        self.n_tasks += 1
        self.quality_sum += quality
        self.total_cost_usd += cost
        self.latency_sum += latency
        self.picks[model] += 1

    @property
    def mean_quality(self) -> float:
        return self.quality_sum / self.n_tasks if self.n_tasks else 0.0

    @property
    def mean_latency_s(self) -> float:
        return self.latency_sum / self.n_tasks if self.n_tasks else 0.0


@dataclass
class ReplaySummary:
    mode: str
    n_tasks: int
    outcomes: dict[str, PolicyOutcome]
    classifier_accuracy: float
    per_task: list[dict]

    @property
    def savings_vs_flagship(self) -> float:
        flag = self.outcomes["flagship"].total_cost_usd
        return 1 - self.outcomes["router"].total_cost_usd / flag if flag else 0.0

    @property
    def quality_delta_vs_flagship(self) -> float:
        return self.outcomes["router"].mean_quality - self.outcomes["flagship"].mean_quality


def _mean_by_task_model(
    rows: Iterable[ResultRow],
) -> dict[tuple[str, str], tuple[float, float, float]]:
    groups: dict[tuple[str, str], list[ResultRow]] = defaultdict(list)
    for r in rows:
        if r.quality is not None and not r.error:
            groups[(r.task_id, r.model_id)].append(r)
    out = {}
    for key, g in groups.items():
        n = len(g)
        out[key] = (
            sum(r.quality for r in g) / n,
            sum(r.cost_usd or 0.0 for r in g) / n,
            sum(r.latency_s or 0.0 for r in g) / n,
        )
    return out


def replay(
    rows: Iterable[ResultRow],
    tasks: Iterable[EvalTask],
    mode: str,
    config: RouterConfig,
    registry: Mapping[str, ModelSpec],
    *,
    leave_one_out: bool = True,
) -> ReplaySummary:
    rows = [r for r in rows if r.quality is not None and not r.error]
    models = sorted({r.model_id for r in rows})
    means = _mean_by_task_model(rows)
    flag = flagship([registry[m] for m in models if m in registry]).id if models else None
    full_board = Leaderboard.from_rows(rows, bootstrap=0)

    outcomes = {name: PolicyOutcome(name) for name in ("router", "flagship", "oracle")}
    outcomes.update({f"fixed:{m}": PolicyOutcome(f"fixed:{m}") for m in models})
    per_task: list[dict] = []
    correct = 0
    for task in tasks:
        if not all((task.id, m) in means for m in models):
            continue
        board = (
            Leaderboard.from_rows([r for r in rows if r.task_id != task.id], bootstrap=0)
            if leave_one_out
            else full_board
        )
        try:
            decision = Router(board, config, registry, models).route(task.prompt, mode)
        except NoDataError:
            continue
        predicted = classify(task.prompt).category
        correct += predicted == task.category

        best_q = max(means[(task.id, m)][0] for m in models)
        oracle = min(
            (m for m in models if means[(task.id, m)][0] >= best_q - 1e-9),
            key=lambda m: means[(task.id, m)][1],
        )
        for name, model in (("router", decision.model_id), ("flagship", flag), ("oracle", oracle)):
            outcomes[name].add(model, *means[(task.id, model)])
        for m in models:
            outcomes[f"fixed:{m}"].add(m, *means[(task.id, m)])
        per_task.append(
            {
                "task_id": task.id,
                "category": task.category,
                "predicted_category": predicted,
                "router_model": decision.model_id,
                "router_quality": means[(task.id, decision.model_id)][0],
                "router_cost_usd": means[(task.id, decision.model_id)][1],
                "flagship_quality": means[(task.id, flag)][0],
                "flagship_cost_usd": means[(task.id, flag)][1],
                "oracle_model": oracle,
            }
        )
    n = len(per_task)
    return ReplaySummary(mode, n, outcomes, correct / n if n else 0.0, per_task)
