"""Re-score stored outputs with the current scorers, without calling any model.

Use after fixing a scorer bug: programmatic scorers rerun for free and earlier judge verdicts
are reused (when the rubric and reference are unchanged) unless a judge is passed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from routewise.evals.framework import (
    EvalTask,
    LLMJudge,
    score_detail_json,
    score_output,
    stored_judge_scores,
)
from routewise.evals.store import ResultRow, ResultStore


async def rescore(
    store: ResultStore,
    tasks: Iterable[EvalTask],
    judge: LLMJudge | None = None,
    only: Callable[[ResultRow, EvalTask], bool] | None = None,
) -> int:
    """Rescore every successful row whose task version is current (and that `only` accepts).
    Returns rows changed."""
    by_key = {(t.id, t.version): t for t in tasks}
    changed = 0
    for row in store.rows():
        task = by_key.get((row.task_id, row.task_version))
        if task is None or row.output is None or (only is not None and not only(row, task)):
            continue
        priors = stored_judge_scores(row.score_detail)
        scored = await score_output(task, row.output, judge, prior_judge_scores=priors)
        detail = score_detail_json(scored.components)
        if scored.value != row.quality or detail != row.score_detail:
            row.quality = scored.value
            row.score_detail = detail
            if judge is not None:
                row.judge_cost_usd = (row.judge_cost_usd or 0.0) + scored.judge_cost_usd
            store.record(row)
            changed += 1
    return changed
