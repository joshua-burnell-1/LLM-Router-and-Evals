"""Markdown tables for the README and CLI."""

from __future__ import annotations

from collections.abc import Iterable

from routewise.router.leaderboard import OVERALL, Leaderboard
from routewise.router.replay import ReplaySummary


def results_table(board: Leaderboard) -> str:
    lines = [
        "| Model | Mean quality (95% CI) | Mean latency | Mean cost / task | Total cost | n |",
        "|---|---|---|---|---|---|",
    ]
    cells = [board.get(m, OVERALL) for m in board.models()]
    for c in sorted((c for c in cells if c), key=lambda c: -c.mean_quality):
        lines.append(
            f"| {c.model_id} | {c.mean_quality:.2f} ({c.ci_low:.2f}-{c.ci_high:.2f}) | "
            f"{c.mean_latency_s:.1f}s | ${c.mean_cost_usd:.5f} | ${c.total_cost_usd:.4f} | {c.n} |"
        )
    return "\n".join(lines)


def category_table(board: Leaderboard) -> str:
    categories = board.categories() + [OVERALL]
    lines = [
        "| Model | " + " | ".join(categories) + " |",
        "|---|" + "---|" * len(categories),
    ]
    for model in board.models():
        values = []
        for cat in categories:
            cell = board.get(model, cat)
            values.append(f"{cell.mean_quality:.2f}" if cell else "-")
        lines.append(f"| {model} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def replay_table(summaries: Iterable[ReplaySummary]) -> str:
    lines = [
        "| Mode | Policy | Mean quality | Total cost | Cost vs flagship | Picks |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        flag_cost = s.outcomes["flagship"].total_cost_usd
        for name in ("router", "flagship", "oracle"):
            o = s.outcomes[name]
            rel = f"{(o.total_cost_usd / flag_cost - 1):+.0%}" if flag_cost else "-"
            picks = ", ".join(f"{m} {n}" for m, n in o.picks.most_common())
            lines.append(
                f"| {s.mode} | {name} | {o.mean_quality:.3f} | ${o.total_cost_usd:.4f} | "
                f"{rel} | {picks} |"
            )
    return "\n".join(lines)
