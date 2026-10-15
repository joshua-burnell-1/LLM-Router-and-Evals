import pytest

from routewise.evals.framework import score_output
from routewise.evals.store import ResultStore
from routewise.models.registry import ModelSpec
from routewise.studio.catalog import CATALOG, build_scorer_spec
from routewise.studio.model import CheckConfig, StudioBenchmark, list_saved, load_saved, save
from routewise.studio.report import html_report
from routewise.studio.run import estimate_cost, run_studio, summarize
from routewise.studio.templates import STARTERS, TEMPLATES
from tests.conftest import FakeChat

REGISTRY = {
    "cheap": ModelSpec(
        id="cheap", name="Cheap", provider="mock", input_price=0.1, output_price=0.5
    ),
    "pricey": ModelSpec(
        id="pricey", name="Pricey", provider="mock", input_price=4, output_price=20
    ),
}


def bench(**kw):
    base = dict(
        name="Invoice totals",
        category="structured_output",
        prompt="What is the total in this invoice? {{invoice}} Reply with only the number.",
        cases=[
            {"invoice": "Total due: $120", "expected": "120"},
            {"invoice": "Amount: 75 USD", "expected": "75"},
        ],
        checks=[CheckConfig("expected_answer", {"mode": "number"})],
    )
    base.update(kw)
    return StudioBenchmark(**base)


# --- catalogue -------------------------------------------------------------------------------


def test_catalog_entries_are_described_for_humans():
    assert {
        "expected_answer",
        "keywords_all",
        "forbidden",
        "word_limit",
        "json_schema",
        "json_fields",
        "regex",
        "keeps_numbers",
        "no_pii",
        "llm_judge",
    } <= set(CATALOG)
    for info in CATALOG.values():
        assert info.label and info.description and info.label[0].isupper()


def test_build_scorer_spec_converts_friendly_values():
    spec = build_scorer_spec("keywords_all", {"words": "refund, order 7731"}, weight=2)
    assert (spec.type, spec.params, spec.weight) == (
        "contains_all",
        {"strings": ["refund", "order 7731"]},
        2,
    )
    assert build_scorer_spec("word_limit", {"max_words": 40}).params == {"max_words": 40}
    assert build_scorer_spec("expected_answer", {"mode": "contains"}).type == "contains_all"


def test_build_scorer_spec_rejects_bad_values():
    with pytest.raises(ValueError, match="schema"):
        build_scorer_spec("json_schema", {"schema": "{not json"})
    with pytest.raises(ValueError, match="rubric"):
        build_scorer_spec("llm_judge", {"rubric": "be nice"})
    with pytest.raises(ValueError, match="words"):
        build_scorer_spec("keywords_all", {"words": " , "})


# --- benchmark definition --------------------------------------------------------------------


def test_variables_and_validation():
    b = bench()
    assert b.variables() == ["invoice"]
    assert b.validate() == []
    assert any("expected" in e for e in bench(cases=[{"invoice": "x"}]).validate())
    assert any("invoice" in e for e in bench(cases=[{"expected": "1"}]).validate())
    assert any("check" in e.lower() for e in bench(checks=[]).validate())
    assert any("prompt" in e.lower() for e in bench(prompt=" ").validate())
    assert any("task type" in e.lower() for e in bench(category="poetry").validate())


async def test_tasks_compile_per_case_and_score_correctly():
    tasks = bench().to_tasks()
    assert [t.id for t in tasks] == ["studio-invoice-totals-1", "studio-invoice-totals-2"]
    assert tasks[0].prompt.startswith("What is the total in this invoice? Total due: $120")
    assert tasks[0].suite == "studio-invoice-totals"
    assert (await score_output(tasks[0], "120", judge=None)).value == 1.0
    assert (await score_output(tasks[1], "120", judge=None)).value == 0.0


async def test_json_fields_check_parses_expected_json_per_case():
    b = bench(
        prompt="Extract: {{email}}",
        cases=[{"email": "inv 7 for $5", "expected": '{"number": "7", "total": 5}'}],
        checks=[CheckConfig("json_fields", {})],
    )
    assert b.validate() == []
    [task] = b.to_tasks()
    good = await score_output(task, '{"number": "7", "total": 5}', judge=None)
    assert good.value == 1.0
    bad = bench(
        prompt="Extract: {{email}}",
        cases=[{"email": "x", "expected": "nope"}],
        checks=[CheckConfig("json_fields", {})],
    )
    assert any("JSON" in e for e in bad.validate())


async def test_keeps_numbers_uses_the_chosen_column_not_the_instructions():
    b = bench(
        prompt="In 2 sentences, summarize: {{text}}",
        cases=[{"text": "Revenue rose 12% to $4.8M."}],
        checks=[CheckConfig("keeps_numbers", {"column": "text"})],
    )
    [task] = b.to_tasks()
    assert (await score_output(task, "Revenue up 12% to 4.8M.", judge=None)).value == 1.0


def test_save_load_round_trip(tmp_path):
    b = bench(description="demo")
    path = save(b, tmp_path)
    assert path.name == "invoice-totals.yaml"
    loaded = load_saved(tmp_path)["invoice-totals"]
    assert loaded == b
    assert [x.name for x in list_saved(tmp_path)] == ["Invoice totals"]


def test_templates_are_valid_and_cover_the_checks():
    assert len(STARTERS) >= 4 and len(TEMPLATES) == len(STARTERS) + 1  # plus "Blank"
    used = set()
    for t in STARTERS:
        assert t.validate() == [], t.name
        used.update(c.key for c in t.checks)
    assert {"llm_judge", "json_fields", "expected_answer", "keeps_numbers"} <= used


# --- cost, run, summary, report -----------------------------------------------------------------


def test_cost_estimate_scales_with_models_and_cases():
    one = estimate_cost(bench(), ["cheap"], REGISTRY)
    both = estimate_cost(bench(), ["cheap", "pricey"], REGISTRY)
    assert one.calls == 2 and both.calls == 4
    assert both.total_usd > one.total_usd > 0
    assert both.per_model["pricey"] > both.per_model["cheap"]


def test_cost_estimate_includes_the_judge():
    judged = bench(checks=[CheckConfig("llm_judge", {"rubric": "5: great\n3: ok\n1: bad"})])
    plain = estimate_cost(bench(), ["cheap"], REGISTRY)
    with_judge = estimate_cost(judged, ["cheap"], REGISTRY, judge_model="pricey")
    assert with_judge.judge_usd > 0 and with_judge.total_usd > plain.total_usd


async def test_run_and_summarize(tmp_path):
    store = ResultStore(tmp_path / "r.db")
    chat = FakeChat(replies={"cheap": "120", "pricey": "120"})
    summary = await run_studio(
        bench(), ["cheap", "pricey"], store, chat_fn=chat, judge=None, progress=lambda _: None
    )
    assert summary.n_done == 4
    res = summarize(bench(), ["cheap", "pricey"], store, REGISTRY)
    by = {m.model_id: m for m in res.models}
    assert by["cheap"].mean_score == 0.5 and by["cheap"].cases == 2
    assert by["cheap"].pass_rate == 0.5
    assert res.best_quality in ("cheap", "pricey")
    assert res.best_value == "cheap"  # same quality, far cheaper
    assert {c.case_index for c in res.cases} == {0, 1}
    assert by["cheap"].checks["Matches the expected answer"] == 0.5


async def test_summary_ignores_results_from_an_edited_prompt(tmp_path):
    store = ResultStore(tmp_path / "r.db")
    await run_studio(
        bench(), ["cheap"], store, chat_fn=FakeChat(), judge=None, progress=lambda _: None
    )
    edited = bench(prompt="Total? {{invoice}}")
    assert summarize(edited, ["cheap"], store, REGISTRY).models == []


async def test_html_report_is_self_contained_and_escaped(tmp_path):
    store = ResultStore(tmp_path / "r.db")
    chat = FakeChat(replies={"cheap": "<script>alert(1)</script>", "pricey": "120"})
    b = bench(name="Invoice <totals>")
    await run_studio(
        b, ["cheap", "pricey"], store, chat_fn=chat, judge=None, progress=lambda _: None
    )
    html = html_report(
        b, summarize(b, ["cheap", "pricey"], store, REGISTRY), name=lambda m: REGISTRY[m].name
    )
    assert html.startswith("<!doctype html>")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Invoice &lt;totals&gt;" in html and "Pricey" in html
    assert "http" not in html.split("<body")[0].lower() or "fonts" not in html  # no remote deps


def test_saved_studio_benchmarks_join_the_eval_set(monkeypatch, tmp_path):
    from routewise.evals.loader import all_tasks

    monkeypatch.setenv("ROUTEWISE_STUDIO_DIR", str(tmp_path))
    save(bench(), tmp_path)
    ids = [t.id for t in all_tasks()]
    assert "studio-invoice-totals-1" in ids and "studio-invoice-totals-2" in ids
    assert "studio-invoice-totals-1" not in [t.id for t in all_tasks(include_studio=False)]
    assert [t.id for t in all_tasks(suites=["studio-invoice-totals"])] == [
        "studio-invoice-totals-1",
        "studio-invoice-totals-2",
    ]


def test_invalid_saved_benchmark_is_skipped_not_fatal(monkeypatch, tmp_path):
    from routewise.evals.loader import all_tasks

    monkeypatch.setenv("ROUTEWISE_STUDIO_DIR", str(tmp_path))
    save(bench(name="Broken", cases=[{"invoice": "x"}]), tmp_path)  # missing expected
    assert not any(t.id.startswith("studio-broken") for t in all_tasks())
