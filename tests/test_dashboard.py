from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from routewise.evals.store import ResultRow, ResultStore
from routewise.evals.tasks import BUILTIN_TASKS
from tests.conftest import SYNTHETIC_REGISTRY_YAML

APP = str(Path(__file__).resolve().parents[1] / "routewise" / "dashboard" / "router_page.py")
PROFILE = {"big": (0.95, 0.010, 3.0), "mid": (0.90, 0.002, 1.5), "small": (0.50, 0.0002, 0.8)}


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(SYNTHETIC_REGISTRY_YAML)
    monkeypatch.setenv("ROUTEWISE_MODELS_CONFIG", str(models))
    db = tmp_path / "results.db"
    monkeypatch.setenv("ROUTEWISE_DB", str(db))
    return db


def _seed(db):
    store = ResultStore(db)
    for task in BUILTIN_TASKS[:8]:
        for model, (q, cost, lat) in PROFILE.items():
            store.record(
                ResultRow(
                    task_id=task.id,
                    task_version=task.version,
                    model_id=model,
                    created_at="2026-10-14T00:00:00+00:00",
                    category=task.category,
                    suite=task.suite,
                    quality=q,
                    cost_usd=cost,
                    latency_s=lat,
                    output="x",
                )
            )


def _md(at):
    return "\n".join(m.value for m in at.markdown)


def test_empty_database_shows_instructions(env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "routewise eval" in _md(at)


def test_landing_explains_the_product_and_shows_evidence(env):
    _seed(env)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    md = _md(at)
    assert "Model router" in md and "Enter a prompt" in md
    assert "Models compared" in md and "Cost saved" in md
    assert at.radio(key="mode").options == ["Best answer", "Best value", "Lowest cost"]
    assert set(at.multiselect(key="models").value) == {"big", "mid", "small"}
    labels = [t.label for t in at.tabs]
    assert "Quality by task type" in labels and "How routing works" in labels


def test_route_shows_verdict_and_every_model(env):
    _seed(env)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="mode").set_value("balanced")
    at.text_area(key="prompt").input("Calculate how many units remain after shipping 15%")
    at.button(key="route").click()
    at.run()
    assert not at.exception
    md = _md(at)
    assert "Send this prompt to mid" in md
    assert "Reasoning" in md
    assert "Recommended" in md and "Flagship" in md
    # every candidate gets a card, including the one excluded by the quality guard
    assert md.count('class="rw-card') >= 3
    assert "Not eligible" in md


def test_empty_prompt_asks_for_input(env):
    _seed(env)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="route").click()
    at.run()
    assert not at.exception
    assert any("Enter a prompt" in w.value for w in at.warning)


def test_markdown_escape_stops_dollar_amounts_rendering_as_latex():
    from routewise.dashboard import escape_markdown

    text = "Winner a ($0.00020/query) over b ($0.00200/query)"
    assert escape_markdown(text) == "Winner a (\\$0.00020/query) over b (\\$0.00200/query)"


def test_demote_headings_keeps_response_headings_small():
    from routewise.dashboard import demote_headings

    text = "# Plan\nbody\n## Scoring\n#### Detail\nnot # a heading"
    assert demote_headings(text) == "##### Plan\nbody\n##### Scoring\n##### Detail\nnot # a heading"


def test_demote_headings_ignores_code_fences():
    from routewise.dashboard import demote_headings

    text = "```python\n# a comment\n```\n# Title"
    assert demote_headings(text) == "```python\n# a comment\n```\n##### Title"
