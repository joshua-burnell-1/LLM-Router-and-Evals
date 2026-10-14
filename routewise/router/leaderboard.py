"""Per-category leaderboard built from stored eval results."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from routewise.evals.framework import EvalTask
from routewise.evals.store import ResultRow, ResultStore

OVERALL = "overall"


@dataclass
class CellStats:
    model_id: str
    category: str
    n: int
    mean_quality: float
    ci_low: float
    ci_high: float
    mean_latency_s: float
    mean_cost_usd: float
    total_cost_usd: float


def bootstrap_ci(
    values: list[float], *, iterations: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    if iterations <= 0 or len(values) < 2 or len(set(values)) == 1:
        mean = sum(values) / len(values)
        return mean, mean
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(iterations))
    lo = means[int(alpha / 2 * iterations)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return lo, hi


class Leaderboard:
    def __init__(self, cells: dict[tuple[str, str], CellStats]):
        self.cells = cells

    @classmethod
    def from_rows(cls, rows: Iterable[ResultRow], *, bootstrap: int = 1000) -> Leaderboard:
        groups: dict[tuple[str, str], list[ResultRow]] = defaultdict(list)
        for row in rows:
            if row.quality is None or row.error:
                continue
            groups[(row.model_id, row.category or OVERALL)].append(row)
            groups[(row.model_id, OVERALL)].append(row)
        cells = {}
        for (model, category), group in groups.items():
            qualities = [r.quality for r in group]
            costs = [r.cost_usd or 0.0 for r in group]
            latencies = [r.latency_s or 0.0 for r in group]
            lo, hi = bootstrap_ci(qualities, iterations=bootstrap)
            cells[(model, category)] = CellStats(
                model_id=model,
                category=category,
                n=len(group),
                mean_quality=sum(qualities) / len(qualities),
                ci_low=lo,
                ci_high=hi,
                mean_latency_s=sum(latencies) / len(latencies),
                mean_cost_usd=sum(costs) / len(costs),
                total_cost_usd=sum(costs),
            )
        return cls(cells)

    @classmethod
    def from_store(
        cls,
        store: ResultStore,
        *,
        models: Iterable[str] | None = None,
        current: Iterable[EvalTask] | None = None,
    ) -> Leaderboard:
        return cls.from_rows(store.rows(models=models, current=current))

    def models(self) -> list[str]:
        return sorted({m for m, _ in self.cells})

    def categories(self) -> list[str]:
        return sorted({c for _, c in self.cells if c != OVERALL})

    def get(self, model_id: str, category: str) -> CellStats | None:
        return self.cells.get((model_id, category))

    def table(self, metric: str = "mean_quality") -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = defaultdict(dict)
        for (model, category), cell in self.cells.items():
            out[model][category] = getattr(cell, metric)
        return dict(out)

    def records(self) -> list[dict]:
        return [
            asdict(c) for c in sorted(self.cells.values(), key=lambda c: (c.model_id, c.category))
        ]
