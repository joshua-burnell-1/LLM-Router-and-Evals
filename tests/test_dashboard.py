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


def test_empty_database_shows_instructions(env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("routewise eval" in i.value for i in at.info)


def test_renders_leaderboard_and_savings(env):
    _seed(env)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.sidebar.radio[0].options == ["quality", "balanced", "cost"]
    assert set(at.sidebar.multiselect[0].value) == {"big", "mid", "small"}
    headers = [h.value for h in at.subheader]
    assert any("Leaderboard" in h for h in headers)
    assert any("savings" in h.lower() for h in headers)


def test_query_shows_routed_vs_flagship(env):
    _seed(env)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("balanced")
    at.text_area[0].input("Calculate how many units remain after shipping 15%")
    at.button[0].click()
    at.run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels.get("Routed model") == "mid"
    assert labels.get("Flagship baseline") == "big"
    assert any("reasoning" in i.value for i in at.info)


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
