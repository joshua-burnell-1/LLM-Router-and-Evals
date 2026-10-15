import pytest

from routewise.bench.core import (
    BUILTIN_DIR,
    Benchmark,
    NotCompleteError,
    create_session,
    generate_outputs,
    history,
    load_benchmarks,
    rate,
    redact_identity,
    reveal,
    save_custom_benchmark,
    session_progress,
)
from routewise.bench.store import BenchStore
from tests.conftest import FakeChat

B1 = Benchmark(id="b1", title="Write a PRD", prompt="Write a PRD for X.", suite="t")
B2 = Benchmark(id="b2", title="Prioritize", prompt="Prioritize this backlog.", suite="t")


@pytest.fixture
def store(tmp_path):
    return BenchStore(tmp_path / "bench.db")


# --- benchmarks ------------------------------------------------------------------------


def test_builtin_pm_benchmarks_ship():
    benches = load_benchmarks(BUILTIN_DIR, custom_dir=None)
    pm = [b for b in benches.values() if b.suite == "pm-core"]
    assert 6 <= len(pm) <= 7
    for b in pm:
        assert len(b.prompt) > 300, b.id  # realistic scenario, not a one-liner
        assert b.guidance, b.id


def test_version_tracks_prompt_only():
    a = Benchmark(id="x", title="t", prompt="p", guidance="g1")
    b = Benchmark(id="x", title="t2", prompt="p", guidance="g2")
    c = Benchmark(id="x", title="t", prompt="p2")
    assert a.version == b.version != c.version


def test_save_and_load_custom_benchmark(tmp_path):
    custom = tmp_path / "custom"
    path = save_custom_benchmark(
        Benchmark(id="", title="My Roadmap Review!", prompt="Review my roadmap", suite="mine"),
        custom,
    )
    assert path.name == "my-roadmap-review.yaml"
    benches = load_benchmarks(tmp_path / "none", custom_dir=custom)
    assert benches["my-roadmap-review"].suite == "mine"


def test_custom_benchmark_ids_are_unique(tmp_path):
    custom = tmp_path / "custom"
    save_custom_benchmark(Benchmark(id="", title="Dup", prompt="p"), custom)
    with pytest.raises(ValueError, match="exists"):
        save_custom_benchmark(Benchmark(id="", title="Dup", prompt="q"), custom)


def test_custom_benchmark_needs_title_and_prompt(tmp_path):
    with pytest.raises(ValueError):
        save_custom_benchmark(Benchmark(id="", title=" ", prompt="p"), tmp_path)
    with pytest.raises(ValueError):
        save_custom_benchmark(Benchmark(id="", title="t", prompt=""), tmp_path)


# --- sessions ------------------------------------------------------------------------------


def test_session_needs_two_models_and_a_listed_baseline(store):
    with pytest.raises(ValueError):
        create_session(store, [B1], ["m1"], baseline="m1")
    with pytest.raises(ValueError):
        create_session(store, [B1], ["m1", "m2"], baseline="m3")


def test_labels_are_shuffled_per_benchmark_and_reproducible(store):
    models = ["m1", "m2", "m3", "m4"]
    benches = [Benchmark(id=f"b{i}", title="t", prompt=f"p{i}") for i in range(8)]
    sid = create_session(store, benches, models, baseline="m1", seed=7)
    outputs = store.outputs(sid)
    assert len(outputs) == 32
    orders = {}
    for o in outputs:
        orders.setdefault(o.benchmark_id, {})[o.label] = o.model_id
    assert all(sorted(v) == ["A", "B", "C", "D"] for v in orders.values())
    # not the same order on every prompt
    assert len({tuple(v[k] for k in "ABCD") for v in orders.values()}) > 1
    sid2 = create_session(store, benches, models, baseline="m1", seed=7)
    again = {(o.benchmark_id, o.label): o.model_id for o in store.outputs(sid2)}
    assert again == {(o.benchmark_id, o.label): o.model_id for o in outputs}


async def test_generate_fills_outputs_and_hides_identity_instruction(store):
    chat = FakeChat(replies={"m1": "one", "m2": "two"})
    sid = create_session(store, [B1, B2], ["m1", "m2"], baseline="m1")
    done = await generate_outputs(store, sid, {"b1": B1, "b2": B2}, chat_fn=chat)
    assert done == 4
    assert {o.output for o in store.outputs(sid)} == {"one", "two"}
    assert len(chat.calls) == 4
    # rerun does nothing: outputs exist
    assert await generate_outputs(store, sid, {"b1": B1, "b2": B2}, chat_fn=chat) == 0


async def test_generate_records_errors_and_retries_them(store):
    chat = FakeChat(fail={"m2": 1})
    sid = create_session(store, [B1], ["m1", "m2"], baseline="m1")
    await generate_outputs(store, sid, {"b1": B1}, chat_fn=chat)
    assert any(o.error for o in store.outputs(sid))
    await generate_outputs(store, sid, {"b1": B1}, chat_fn=chat)
    assert all(o.output and not o.error for o in store.outputs(sid))


async def _ready_session(store, scores):
    """scores: {model: [score per benchmark]} for B1, B2 with models m1 (baseline), m2."""
    sid = create_session(store, [B1, B2], ["m1", "m2"], baseline="m1", seed=1)
    await generate_outputs(store, sid, {"b1": B1, "b2": B2}, chat_fn=FakeChat())
    for o in store.outputs(sid):
        idx = 0 if o.benchmark_id == "b1" else 1
        rate(
            store,
            o.id,
            scores[o.model_id][idx],
            liked=f"{o.model_id} good",
            disliked=f"{o.model_id} bad",
        )
    return sid


def test_rating_validation(store):
    sid = create_session(store, [B1], ["m1", "m2"], baseline="m1")
    oid = store.outputs(sid)[0].id
    with pytest.raises(ValueError):
        rate(store, oid, 6, liked="x")
    with pytest.raises(ValueError):
        rate(store, oid, 3)  # needs a liked or disliked comment
    with pytest.raises(ValueError, match="no output"):
        rate(store, oid, 3, liked="x")  # nothing generated yet


async def test_reveal_refused_until_everything_is_rated(store):
    sid = create_session(store, [B1], ["m1", "m2"], baseline="m1")
    await generate_outputs(store, sid, {"b1": B1}, chat_fn=FakeChat())
    first = store.outputs(sid)[0]
    rate(store, first.id, 4, liked="clear")
    assert session_progress(store, sid) == (1, 2)
    with pytest.raises(NotCompleteError):
        reveal(store, sid)


async def test_reveal_shows_preferred_model_and_baseline_delta(store):
    sid = await _ready_session(store, {"m1": [3, 3], "m2": [5, 4]})
    result = reveal(store, sid)
    assert result.preferred == "m2"
    by_model = {m.model_id: m for m in result.models}
    assert by_model["m2"].mean_score == 4.5 and by_model["m2"].wins == 2
    assert by_model["m2"].delta_vs_baseline == pytest.approx(1.5)
    assert by_model["m1"].is_baseline and by_model["m1"].delta_vs_baseline == 0
    assert "m2 good" in by_model["m2"].liked and "m1 bad" in by_model["m1"].disliked
    assert {(r.benchmark_id, r.model_id): r.score for r in result.rows}[("b1", "m2")] == 5
    assert store.session(sid).revealed_at


async def test_ties_split_wins(store):
    sid = await _ready_session(store, {"m1": [4, 2], "m2": [4, 5]})
    by_model = {m.model_id: m for m in reveal(store, sid).models}
    assert by_model["m1"].wins == 0.5 and by_model["m2"].wins == 1.5


async def test_history_aggregates_revealed_sessions_on_same_prompt_version(store):
    await _ready_session(store, {"m1": [3, 3], "m2": [5, 5]})
    s2 = await _ready_session(store, {"m1": [4, 4], "m2": [2, 2]})
    reveal(store, store.sessions()[0].session_id)
    reveal(store, s2)
    rows = {(h.model_id): h for h in history(store, {"b1": B1, "b2": B2})}
    assert rows["m2"].sessions == 2 and rows["m2"].mean_score == pytest.approx(3.5)
    # unrevealed sessions never leak into history
    await _ready_session(store, {"m1": [1, 1], "m2": [1, 1]})
    rows = {(h.model_id): h for h in history(store, {"b1": B1, "b2": B2})}
    assert rows["m2"].sessions == 2


async def test_history_ignores_results_from_an_edited_prompt(store):
    await _ready_session(store, {"m1": [3, 3], "m2": [5, 5]})
    reveal(store, store.sessions()[0].session_id)
    edited = Benchmark(id="b1", title="Write a PRD", prompt="Write a PRD for Y.", suite="t")
    rows = {h.model_id: h for h in history(store, {"b1": edited, "b2": B2})}
    assert rows["m2"].ratings == 1  # only b2 still matches


@pytest.mark.parametrize(
    "text",
    [
        "As Claude, I think",
        "I'm ChatGPT",
        "OpenAI's guidance",
        "Anthropic built me",
        "Gemini here",
        "gpt-6-astra says",
    ],
)
def test_redact_identity(text):
    assert "[model]" in redact_identity(text)


def test_redact_leaves_ordinary_text():
    assert redact_identity("Ship the MVP in Q3.") == "Ship the MVP in Q3."


def test_custom_dir_can_be_redirected_by_env(tmp_path, monkeypatch):
    from routewise.bench.core import custom_dir

    monkeypatch.setenv("ROUTEWISE_CUSTOM_BENCHMARKS", str(tmp_path / "mine"))
    assert custom_dir() == tmp_path / "mine"
    save_custom_benchmark(Benchmark(id="", title="Env Bench", prompt="p"))
    assert (tmp_path / "mine" / "env-bench.yaml").exists()
    assert "env-bench" in load_benchmarks()
