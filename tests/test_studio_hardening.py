"""Regression tests for the studio review findings (bad input must never break other pages)."""

import yaml

from routewise.evals.framework import LLMJudge
from routewise.evals.loader import all_tasks, studio_tasks
from routewise.evals.store import ResultStore
from routewise.studio.model import CheckConfig, StudioBenchmark, load_saved, save, slugify
from routewise.studio.run import estimate_cost, run_studio, summarize
from tests.conftest import FakeChat
from tests.test_studio import REGISTRY, bench

JUDGE_BAD = '{"score": 1, "reason": "bad"}'
JUDGE_GOOD = '{"score": 5, "reason": "great"}'
RUBRIC = "5: great\n3: ok\n1: bad"


def _quiet(_):
    return None


# --- 1 and 2: only the catalogue's own placeholders are filled in --------------------------------


def test_judge_shown_the_expected_answer_requires_an_expected_column():
    b = bench(
        cases=[{"invoice": "Total due: $120"}],
        checks=[CheckConfig("llm_judge", {"rubric": RUBRIC, "show_expected": True})],
    )
    assert any("expected answer" in p for p in b.validate())


def test_braces_typed_into_a_check_are_kept_literally():
    b = bench(
        checks=[
            CheckConfig("forbidden", {"words": "{{customer_name}}, {{invoice}}"}),
            CheckConfig("regex", {"pattern": r"\d{2}"}),
        ]
    )
    assert b.validate() == []
    strings = b.to_tasks()[0].scoring[0].params["strings"]
    assert strings == ["{{customer_name}}", "{{invoice}}"]


def test_a_benchmark_that_cannot_compile_is_reported_not_raised(monkeypatch):
    b = bench(prompt="Total? {{invoice}}")
    b.cases = [{"invoice": "x", "expected": "1"}]
    monkeypatch.setattr(
        StudioBenchmark, "to_tasks", lambda self: (_ for _ in ()).throw(KeyError("x"))
    )
    assert any("could not be prepared" in p for p in b.validate())


# --- 3: malformed saved files are skipped ---------------------------------------------------------


def test_malformed_saved_files_are_skipped_everywhere(tmp_path, monkeypatch):
    d = tmp_path / "studio"
    monkeypatch.setenv("ROUTEWISE_STUDIO_DIR", str(d))
    save(bench())
    (d / "broken.yaml").write_text("name: [unclosed")
    (d / "list.yaml").write_text("- a\n- b\n")
    (d / "extra.yaml").write_text(yaml.safe_dump({**bench(name="Extra").to_dict(), "version": 2}))
    (d / "numeric.yaml").write_text(yaml.safe_dump({**bench().to_dict(), "name": 5}))
    (d / "weird-check.yaml").write_text(
        yaml.safe_dump(
            {**bench(name="Weird").to_dict(), "checks": [{"key": "nope"}, {"key": "regex"}]}
        )
    )
    errors: list[str] = []
    saved = load_saved(errors=errors)
    assert {"invoice-totals", "extra", "5", "weird"} <= set(saved)
    assert len(errors) == 2  # the syntax error and the list
    assert all_tasks(include_studio=True)  # nothing raises
    assert all(t.suite != "studio-weird" for t in studio_tasks())  # invalid ones skipped


# --- 4: judge checks run on answers that were already stored --------------------------------------


async def test_adding_a_judge_later_judges_the_stored_answers(tmp_path):
    store = ResultStore(tmp_path / "r.db")
    chat = FakeChat(replies={"cheap": "120", "judge": JUDGE_BAD})
    await run_studio(bench(), ["cheap"], store, chat_fn=chat, judge=None, progress=_quiet)

    judged = bench(
        checks=[
            CheckConfig("expected_answer", {"mode": "number"}),
            CheckConfig("llm_judge", {"rubric": RUBRIC}, required=True),
        ]
    )
    est = estimate_cost(judged, ["cheap"], REGISTRY, judge_model="cheap", store=store)
    assert est.calls == 0 and est.judge_calls == 2 and est.judge_usd > 0

    judge = LLMJudge("judge", chat_fn=chat)
    await run_studio(judged, ["cheap"], store, chat_fn=chat, judge=judge, progress=_quiet)
    assert sum(1 for m, _ in chat.calls if m == "judge") == 2
    res = summarize(judged, ["cheap"], store, REGISTRY)
    assert res.models[0].mean_score == 0.0  # required judge check failed
    assert res.pending_judge == 0


async def test_editing_the_rubric_rejudges_and_flags_stale_verdicts(tmp_path):
    store = ResultStore(tmp_path / "r.db")
    chat = FakeChat(replies={"cheap": "120", "judge": JUDGE_GOOD})
    judge = LLMJudge("judge", chat_fn=chat)
    first = bench(checks=[CheckConfig("llm_judge", {"rubric": RUBRIC})])
    await run_studio(first, ["cheap"], store, chat_fn=chat, judge=judge, progress=_quiet)
    assert summarize(first, ["cheap"], store, REGISTRY).models[0].mean_score == 1.0

    edited = bench(checks=[CheckConfig("llm_judge", {"rubric": RUBRIC + "\n2: meh"})])
    res = summarize(edited, ["cheap"], store, REGISTRY)
    assert res.pending_judge == 2  # old verdicts are not passed off as current

    chat.replies["judge"] = JUDGE_BAD
    await run_studio(edited, ["cheap"], store, chat_fn=chat, judge=judge, progress=_quiet)
    res = summarize(edited, ["cheap"], store, REGISTRY)
    assert res.pending_judge == 0 and res.models[0].mean_score == 0.0


# --- 5: slug collisions never overwrite a different benchmark -------------------------------------


def test_saving_a_different_benchmark_with_the_same_slug_is_refused(tmp_path):
    save(bench(name="Invoice totals"), tmp_path)
    save(bench(name="Invoice totals"), tmp_path)  # same benchmark: fine
    try:
        save(bench(name="Invoice Totals!"), tmp_path)
    except ValueError as exc:
        assert "already" in str(exc)
    else:
        raise AssertionError("expected a refusal")
    assert load_saved(tmp_path)["invoice-totals"].name == "Invoice totals"


def test_non_latin_names_get_distinct_slugs():
    a, b = slugify("請求書の合計"), slugify("Сводка")
    assert a != b and a.startswith("benchmark-") and b.startswith("benchmark-")
    assert slugify("") == "benchmark"


# --- 8 and 12: results reflect the current checks, one sample per case ----------------------------


async def test_results_follow_edited_checks_without_a_rerun(tmp_path):
    store = ResultStore(tmp_path / "r.db")
    chat = FakeChat(replies={"cheap": "120"})
    await run_studio(bench(), ["cheap"], store, chat_fn=chat, judge=None, progress=_quiet)
    assert summarize(bench(), ["cheap"], store, REGISTRY).models[0].mean_score == 0.5

    wordy = bench(checks=[CheckConfig("word_limit", {"max_words": 5})])
    res = summarize(wordy, ["cheap"], store, REGISTRY)
    assert res.models[0].mean_score == 1.0
    assert {c.label for c in res.cases[0].checks} == {"Stays within a word limit"}


async def test_repeat_samples_do_not_count_as_extra_cases(tmp_path):
    store = ResultStore(tmp_path / "r.db")
    chat = FakeChat(replies={"cheap": "120"})
    await run_studio(bench(), ["cheap"], store, chat_fn=chat, judge=None, progress=_quiet)
    row = store.rows()[0]
    row.sample_idx = 1
    store.record(row)
    res = summarize(bench(), ["cheap"], store, REGISTRY)
    assert res.models[0].cases == 2 and len(res.cases) == 2


# --- 10: studio tasks stay out of eval, gate and snapshot by default -----------------------------


def test_studio_tasks_are_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTEWISE_STUDIO_DIR", str(tmp_path))
    save(bench())
    assert not any(t.suite.startswith("studio-") for t in all_tasks())
    assert any(t.suite.startswith("studio-") for t in all_tasks(include_studio=True))


# --- 11: an empty expected JSON object is rejected -----------------------------------------------


def test_empty_expected_json_is_rejected():
    b = bench(
        cases=[{"invoice": "x", "expected": "{}"}],
        checks=[CheckConfig("json_fields", {})],
    )
    assert any("at least one field" in p for p in b.validate())
