from collections import Counter

import pytest

from routewise.evals.framework import SCORERS, score_output
from routewise.evals.tasks import BUILTIN_TASKS

TASK_IDS = [t.id for t in BUILTIN_TASKS]


def test_sixteen_tasks_with_unique_ids():
    assert len(BUILTIN_TASKS) == 16
    assert len(set(TASK_IDS)) == 16


def test_category_counts_match_spec():
    counts = Counter(t.category for t in BUILTIN_TASKS)
    assert counts == {
        "structured_output": 4,
        "reasoning": 4,
        "summarization": 3,
        "voice": 3,
        "safety": 2,
    }


@pytest.mark.parametrize("task", BUILTIN_TASKS, ids=TASK_IDS)
def test_every_task_has_a_programmatic_scorer(task):
    assert any(s.type in SCORERS for s in task.scoring), task.id


@pytest.mark.parametrize("task", BUILTIN_TASKS, ids=TASK_IDS)
def test_judge_components_carry_their_rubric(task):
    for spec in task.scoring:
        if spec.type == "llm_judge":
            rubric = spec.params["rubric"]
            for anchor in ("5", "3", "1"):
                assert f"{anchor}:" in rubric, (task.id, anchor)


@pytest.mark.parametrize("task", BUILTIN_TASKS, ids=TASK_IDS)
async def test_reference_output_scores_perfectly(task):
    assert task.reference_output, task.id
    result = await score_output(task, task.reference_output, judge=None)
    assert result.value == pytest.approx(1.0), (task.id, result.reason)


@pytest.mark.parametrize("task", BUILTIN_TASKS, ids=TASK_IDS)
async def test_empty_output_scores_below_one(task):
    result = await score_output(task, "", judge=None)
    assert result.value < 1.0, task.id
