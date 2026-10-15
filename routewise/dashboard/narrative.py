"""Plain-language text for the dashboard: what was decided, why, and what it means."""

from __future__ import annotations

from collections.abc import Callable

from routewise.bench.core import Reveal
from routewise.router.classifier import Classification
from routewise.router.router import Candidate

CATEGORY_LABELS = {
    "structured_output": "Structured output",
    "reasoning": "Reasoning",
    "summarization": "Summarization",
    "voice": "Voice",
    "safety": "Safety",
    "general": "General",
    "overall": "All tasks",
}

CATEGORY_BLURBS = {
    "structured_output": "JSON, tables, extraction and function-call arguments",
    "reasoning": "maths, logic puzzles, debugging and finding flaws",
    "summarization": "summaries, TL;DRs and action items that keep every fact",
    "voice": "speech transcripts, interruptions and customer replies",
    "safety": "prompt injection, secrets and personal data",
    "general": "anything that does not fit one task type",
}

MODES = {
    "quality": ("Best answer", "Highest measured quality. Cost only breaks ties."),
    "balanced": ("Best value", "Big savings, never more than 0.10 below the best quality."),
    "cost": ("Lowest cost", "Cheapest model within 0.15 of the best quality."),
}


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.replace("_", " ").capitalize())


def detection_note(c: Classification) -> str:
    """How the task type was recognised, in one phrase."""
    if c.method == "rules" and c.matched:
        words = ", ".join(f"“{m}”" for m in c.matched[:3])
        return f"Recognised from the words {words}."
    if c.method == "knn":
        return "Recognised because it is similar to tasks in the eval set."
    if c.method == "llm":
        return "Recognised by a classifier model."
    return "No clear task type, so results across all task types were used."


def verdict(
    chosen: Candidate,
    flagship: Candidate,
    category: str,
    mode: str,
    name: Callable[[str], str | None],
) -> tuple[str, str]:
    """(headline, one-sentence reason) for the routing decision."""

    def nm(model_id: str) -> str:
        return name(model_id) or model_id

    head = f"Send this prompt to {nm(chosen.model_id)}"
    task = f"{category_label(category).lower()} tasks"
    if chosen.model_id == flagship.model_id:
        return head, (
            f"It has the highest expected quality on {task}, so in "
            f"{MODES.get(mode, (mode,))[0].lower()} mode it is worth the flagship price."
        )
    ratio = chosen.mean_cost_usd / flagship.mean_cost_usd if flagship.mean_cost_usd else 1.0
    cost = f"{ratio:.0%} of the cost" if ratio < 1 else f"{ratio:.1f}x the cost"
    dq = chosen.quality - flagship.quality
    if abs(dq) < 0.005:
        quality = f"Same expected quality as {nm(flagship.model_id)} on {task}"
    elif dq < 0:
        quality = f"{-dq:.2f} lower expected quality than {nm(flagship.model_id)} on {task}"
    else:
        quality = f"{dq:.2f} higher expected quality than {nm(flagship.model_id)} on {task}"
    return head, f"{quality}, at {cost}."


def _num(x: float) -> str:
    return f"{x:g}"


def tied_top(result: Reveal, eps: float = 1e-9) -> list[str]:
    """Models sharing first place on both mean score and tasks won."""
    best = result.models[0]
    return [
        m.model_id
        for m in result.models
        if abs(m.mean_score - best.mean_score) < eps and abs(m.wins - best.wins) < eps
    ]


def reveal_headline(
    result: Reveal, name: Callable[[str], str | None], n_tasks: int
) -> tuple[str, str]:
    """(headline, sentence) for a revealed blind session."""

    def nm(model_id: str) -> str:
        return name(model_id) or model_id

    by_id = {m.model_id: m for m in result.models}
    winner = by_id[result.preferred]
    base = by_id[result.baseline]
    leaders = tied_top(result)
    if len(leaders) > 1:
        names = [nm(m) for m in leaders]
        joined = ", ".join(names[:-1]) + f" and {names[-1]}"
        return f"A tie between {joined}", (
            f"Each averaged {winner.mean_score:.1f} out of 5 and won {_num(winner.wins)} of "
            f"{n_tasks} tasks. Rate more tasks to separate them."
        )
    if result.preferred == result.baseline:
        runner = next((m for m in result.models if m.model_id != result.baseline), None)
        gap = (
            f", {winner.mean_score - runner.mean_score:.1f} ahead of {nm(runner.model_id)}"
            if runner
            else ""
        )
        return "Your current model held its ground", (
            f"{nm(winner.model_id)} averaged {winner.mean_score:.1f} out of 5{gap}, and won "
            f"{_num(winner.wins)} of {n_tasks} tasks."
        )
    diff = winner.mean_score - base.mean_score
    return f"You preferred {nm(winner.model_id)}", (
        f"It averaged {winner.mean_score:.1f} out of 5, {diff:.1f} higher than "
        f"{nm(base.model_id)} (your current model), and won {_num(winner.wins)} of "
        f"{n_tasks} tasks."
    )
