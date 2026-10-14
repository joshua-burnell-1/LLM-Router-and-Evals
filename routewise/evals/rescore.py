"""Re-score stored outputs with the current scorers, without calling any model.

Use after fixing a scorer bug: programmatic scorers rerun for free and earlier judge verdicts
are reused unless a judge is passed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from routewise.evals.framework import EvalTask, LLMJudge, Score, score_output
from routewise.evals.store import ResultStore


async def rescore(
    store: ResultStore, tasks: Iterable[EvalTask], judge: LLMJudge | None = None
) -> int:
    """Rescore every successful row whose task version is current. Returns rows changed."""
    by_key = {(t.id, t.version): t for t in tasks}
    changed = 0
    for row in store.rows():
        task = by_key.get((row.task_id, row.task_version))
        if task is None or row.output is None:
            continue
        old = json.loads(row.score_detail or "[]")
        priors = [
            Score(d["value"], d["reason"], "llm_judge") for d in old if d["scorer"] == "llm_judge"
        ]
        scored = await score_output(task, row.output, judge, prior_judge_scores=priors)
        detail = json.dumps(
            [{"scorer": c.scorer, "value": c.value, "reason": c.reason} for c in scored.components]
        )
        if scored.value != row.quality or detail != row.score_detail:
            row.quality = scored.value
            row.score_detail = detail
            if judge is not None:
                row.judge_cost_usd = (row.judge_cost_usd or 0.0) + scored.judge_cost_usd
            store.record(row)
            changed += 1
    return changed
