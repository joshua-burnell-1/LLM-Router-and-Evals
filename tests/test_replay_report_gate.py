import json

import pytest

from routewise.evals.rescore import rescore
from routewise.evals.store import ResultRow, ResultStore
from routewise.router.gate import gate, load_snapshot, save_snapshot, snapshot
from routewise.router.leaderboard import Leaderboard
from routewise.router.replay import replay
from routewise.router.report import category_table, replay_table, results_table
from routewise.router.router import ModeWeights, RouterConfig
from tests.conftest import make_task, synthetic_rows, synthetic_tasks

CONFIG = RouterConfig(
    modes={
        "quality": ModeWeights(quality=1.0, cost=0.005, latency=0.005),
        "balanced": ModeWeights(quality=1.0, cost=0.25, latency=0.10, max_quality_drop=0.10),
    },
    min_samples=2,
)


# --- replay ---------------------------------------------------------------------------


def test_replay_balanced_saves_versus_flagship(synthetic_registry):
    summary = replay(synthetic_rows(), synthetic_tasks(), "balanced", CONFIG, synthetic_registry)
    router, flag = summary.outcomes["router"], summary.outcomes["flagship"]
    assert summary.n_tasks == 8
    assert router.picks == {"mid": 4, "small": 4}
    assert router.total_cost_usd == pytest.approx(4 * 0.002 + 4 * 0.0002)
    assert flag.total_cost_usd == pytest.approx(8 * 0.010)
    assert summary.savings_vs_flagship == pytest.approx(1 - 0.0088 / 0.08)
    assert summary.quality_delta_vs_flagship == pytest.approx((4 * 0.90 + 4 * 0.93) / 8 - 0.95)
    assert summary.classifier_accuracy == 1.0
    assert {"oracle", "fixed:big", "fixed:mid", "fixed:small"} <= set(summary.outcomes)


def test_replay_quality_mode_matches_flagship(synthetic_registry):
    summary = replay(synthetic_rows(), synthetic_tasks(), "quality", CONFIG, synthetic_registry)
    assert summary.outcomes["router"].picks == {"big": 8}
    assert summary.savings_vs_flagship == pytest.approx(0.0)


def test_replay_is_leave_one_out(synthetic_registry):
    # small scores 1.0 on reasoning-0 and 0.82 elsewhere. In-sample, its reasoning mean
    # (0.865) passes the quality guard and beats mid; leave-one-out, reasoning-0 cannot vouch
    # for itself (mean 0.82 fails the guard), so the router picks mid for it.
    rows = []
    for r in synthetic_rows():
        if r.model_id == "small" and r.category == "reasoning":
            r.quality = 1.0 if r.task_id == "reasoning-0" else 0.82
        rows.append(r)
    in_sample = replay(
        rows, synthetic_tasks(), "balanced", CONFIG, synthetic_registry, leave_one_out=False
    )
    assert {t["task_id"]: t["router_model"] for t in in_sample.per_task}["reasoning-0"] == "small"
    summary = replay(rows, synthetic_tasks(), "balanced", CONFIG, synthetic_registry)
    per_task = {t["task_id"]: t["router_model"] for t in summary.per_task}
    assert per_task["reasoning-0"] == "mid"


def test_replay_oracle_picks_cheapest_perfect_model(synthetic_registry):
    rows = synthetic_rows() + [
        ResultRow(
            task_id="extra",
            task_version="v1",
            model_id=m,
            created_at="x",
            category="reasoning",
            quality=q,
            cost_usd=c,
            latency_s=1.0,
        )
        for m, q, c in [("big", 1.0, 0.01), ("mid", 1.0, 0.002), ("small", 0.0, 0.0002)]
    ]
    tasks = synthetic_tasks() + [make_task("extra", "reasoning", "Calculate the extra")]
    summary = replay(rows, tasks, "balanced", CONFIG, synthetic_registry)
    oracle = {t["task_id"]: t["oracle_model"] for t in summary.per_task}
    assert oracle["extra"] == "mid"


# --- report -----------------------------------------------------------------------------


def test_results_table_has_quality_latency_cost():
    md = results_table(Leaderboard.from_rows(synthetic_rows()))
    header = md.splitlines()[0]
    for col in ("Model", "Mean quality", "Mean latency", "Total cost"):
        assert col in header
    assert "| big |" in md and "0.95" in md


def test_category_table_is_model_by_category():
    md = category_table(Leaderboard.from_rows(synthetic_rows()))
    assert md.splitlines()[0].startswith("| Model | reasoning | safety")
    assert "| small | 0.60 | 0.93" in md


def test_replay_table(synthetic_registry):
    summary = replay(synthetic_rows(), synthetic_tasks(), "balanced", CONFIG, synthetic_registry)
    md = replay_table([summary])
    assert "balanced" in md and "router" in md and "flagship" in md and "%" in md


# --- snapshot and gate ------------------------------------------------------------------


def test_gate_passes_against_own_snapshot(tmp_path):
    board = Leaderboard.from_rows(synthetic_rows())
    path = tmp_path / "baseline.json"
    save_snapshot(snapshot(board), path)
    result = gate(board, load_snapshot(path), threshold=0.05)
    assert result.passed and result.regressions == []
    assert json.loads(path.read_text())["models"]["big"]["reasoning"]["mean_quality"] == 0.95


def test_gate_flags_quality_drop_beyond_threshold():
    baseline = snapshot(Leaderboard.from_rows(synthetic_rows()))
    worse = []
    for row in synthetic_rows():
        if row.model_id == "mid" and row.category == "reasoning":
            row.quality = 0.70
        worse.append(row)
    result = gate(Leaderboard.from_rows(worse), baseline, threshold=0.05)
    assert not result.passed
    [reg] = [r for r in result.regressions if r.category == "reasoning"]
    assert (reg.model_id, round(reg.drop, 2)) == ("mid", 0.20)


def test_gate_ignores_drops_within_threshold():
    baseline = snapshot(Leaderboard.from_rows(synthetic_rows()))
    rows = synthetic_rows()
    for row in rows:
        if row.model_id == "big":
            row.quality -= 0.03
    assert gate(Leaderboard.from_rows(rows), baseline, threshold=0.05).passed


# --- rescore ----------------------------------------------------------------------------


async def test_rescore_updates_quality_and_keeps_judge_component(tmp_path):
    from routewise.evals.framework import EvalTask, ScorerSpec

    task = EvalTask(
        id="t",
        category="reasoning",
        prompt="2+2?",
        scoring=[
            ScorerSpec("exact_match", {"expected": 4, "extract": "number"}),
            ScorerSpec("llm_judge", {"rubric": "5: a\n3: b\n1: c"}),
        ],
    )
    store = ResultStore(tmp_path / "r.db")
    store.record(
        ResultRow(
            task_id="t",
            task_version=task.version,
            model_id="m",
            created_at="x",
            category="reasoning",
            quality=0.25,
            output="the answer is 4",
            score_detail=json.dumps(
                [
                    {"scorer": "exact_match", "value": 0.0, "reason": "old bug"},
                    {"scorer": "llm_judge", "value": 0.5, "reason": "judge 3/5"},
                ]
            ),
        )
    )
    changed = await rescore(store, [task])
    assert changed == 1
    [row] = store.rows()
    assert row.quality == pytest.approx(0.75)
    detail = json.loads(row.score_detail)
    assert [d["value"] for d in detail] == [1.0, 0.5]


def test_replay_classifier_is_leave_one_out(synthetic_registry):
    # Two tasks with identical text but different labels: each can only be predicted from
    # the other, so leave-one-out accuracy on them is 0 even though in-sample would be 1.
    twins = [
        make_task("twin-a", "safety", "zorbly widget alignment quux"),
        make_task("twin-b", "reasoning", "zorbly widget alignment quux"),
    ]
    rows = synthetic_rows() + [
        ResultRow(
            task_id=t.id,
            task_version="v1",
            model_id=m,
            created_at="x",
            category=t.category,
            quality=0.9,
            cost_usd=0.001,
            latency_s=1.0,
        )
        for t in twins
        for m in ("big", "mid", "small")
    ]
    summary = replay(rows, twins, "balanced", CONFIG, synthetic_registry)
    predicted = {t["task_id"]: t["predicted_category"] for t in summary.per_task}
    assert predicted == {"twin-a": "reasoning", "twin-b": "safety"}
    assert summary.classifier_accuracy == 0.0
