import textwrap

import pytest

from routewise.evals.runner import main, run_eval
from routewise.evals.store import ResultStore
from tests.conftest import FakeChat, make_task

TASKS = [make_task("t1"), make_task("t2", prompt="3+1?")]


@pytest.fixture
def store(tmp_path):
    return ResultStore(tmp_path / "results.db")


async def test_writes_one_row_per_task_and_model(store):
    chat = FakeChat(replies={"good": "4", "bad": "5"})
    summary = await run_eval(
        TASKS, ["good", "bad"], store, judge=None, chat_fn=chat, progress=lambda _: None
    )
    rows = store.rows()
    assert len(rows) == 4 and summary.n_done == 4
    by_model = {(r.task_id, r.model_id): r for r in rows}
    good = by_model[("t1", "good")]
    assert good.quality == 1.0 and by_model[("t1", "bad")].quality == 0.0
    assert (good.input_tokens, good.output_tokens, good.latency_s) == (100, 20, 0.5)
    assert good.cost_usd == 0.01 and good.output == "4" and good.created_at
    assert good.category == "reasoning" and good.task_version == TASKS[0].version
    assert summary.total_cost_usd == pytest.approx(0.04)


async def test_rerun_is_idempotent(store):
    chat = FakeChat()
    await run_eval(TASKS, ["m"], store, judge=None, chat_fn=chat, progress=lambda _: None)
    summary = await run_eval(TASKS, ["m"], store, judge=None, chat_fn=chat, progress=lambda _: None)
    assert len(chat.calls) == 2
    assert summary.n_skipped == 2 and summary.n_done == 0
    assert len(store.rows()) == 2


async def test_force_reruns_and_replaces(store):
    chat = FakeChat()
    await run_eval(TASKS, ["m"], store, judge=None, chat_fn=chat, progress=lambda _: None)
    await run_eval(
        TASKS, ["m"], store, judge=None, chat_fn=chat, force=True, progress=lambda _: None
    )
    assert len(chat.calls) == 4
    assert len(store.rows()) == 2


async def test_errors_are_recorded_then_retried(store):
    chat = FakeChat(fail={"m": 1})
    first = await run_eval(
        TASKS[:1], ["m"], store, judge=None, chat_fn=chat, progress=lambda _: None
    )
    assert first.n_errors == 1
    assert store.rows() == []
    [err] = store.rows(include_errors=True)
    assert "exploded" in err.error and err.quality is None
    second = await run_eval(
        TASKS[:1], ["m"], store, judge=None, chat_fn=chat, progress=lambda _: None
    )
    assert second.n_done == 1
    assert [r.quality for r in store.rows(include_errors=True)] == [1.0]


async def test_repeats_create_samples(store):
    await run_eval(
        TASKS[:1], ["m"], store, repeats=3, judge=None, chat_fn=FakeChat(), progress=lambda _: None
    )
    assert sorted(r.sample_idx for r in store.rows()) == [0, 1, 2]


async def test_budget_cap_stops_early(store):
    many = [make_task(f"t{i}") for i in range(10)]
    summary = await run_eval(
        many,
        ["m"],
        store,
        concurrency=1,
        max_cost=0.025,
        judge=None,
        chat_fn=FakeChat(cost=0.01),
        progress=lambda _: None,
    )
    assert summary.stopped_for_budget
    assert summary.n_done == 3  # 0.00 -> 0.01 -> 0.02 -> 0.03 >= cap
    assert len(store.rows()) == 3


async def test_changed_prompt_gets_new_row_and_current_filter(store):
    chat = FakeChat()
    old = make_task("t1", prompt="2+2?")
    new = make_task("t1", prompt="Compute 2+2.")
    await run_eval([old], ["m"], store, judge=None, chat_fn=chat, progress=lambda _: None)
    await run_eval([new], ["m"], store, judge=None, chat_fn=chat, progress=lambda _: None)
    assert len(chat.calls) == 2
    assert len(store.rows()) == 2
    assert [r.task_version for r in store.rows(current=[new])] == [new.version]


async def test_progress_line_per_job(store):
    lines = []
    await run_eval(TASKS, ["m"], store, judge=None, chat_fn=FakeChat(), progress=lines.append)
    assert len(lines) == 2
    assert lines[0].startswith("[1/2]") and "t1" in lines[0] and "q=1.00" in lines[0]


def test_cli_limit_and_models(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models.yaml"
    models.write_text(
        textwrap.dedent("""
        models:
          - {id: m1, provider: mock, input_price: 1, output_price: 2}
          - {id: m2, provider: mock, input_price: 1, output_price: 2}
    """)
    )
    monkeypatch.setenv("ROUTEWISE_MODELS_CONFIG", str(models))
    db = tmp_path / "r.db"
    code = main(["--limit", "2", "--models", "m1", "--db", str(db), "--no-judge"])
    assert code == 0
    rows = ResultStore(db).rows(include_errors=True)
    assert len(rows) == 2 and {r.model_id for r in rows} == {"m1"}
    assert "[2/2]" in capsys.readouterr().out
