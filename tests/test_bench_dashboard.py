import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from routewise.bench.store import BenchStore

PAGE = str(Path(__file__).resolve().parents[1] / "routewise" / "dashboard" / "bench_page.py")
APP = str(Path(__file__).resolve().parents[1] / "routewise" / "dashboard" / "app.py")


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(
        textwrap.dedent("""
        models:
          - {id: m1, provider: mock, input_price: 1, output_price: 2}
          - {id: m2, provider: mock, input_price: 1, output_price: 2}
    """)
    )
    monkeypatch.setenv("ROUTEWISE_MODELS_CONFIG", str(models))
    monkeypatch.setenv("ROUTEWISE_DB", str(tmp_path / "r.db"))
    monkeypatch.setenv("ROUTEWISE_CUSTOM_BENCHMARKS", str(tmp_path / "custom"))
    return tmp_path


def _md(at):
    return "\n".join(m.value for m in at.markdown)


def _start_one_task_session(at):
    for cb in at.checkbox:
        if cb.key and cb.key.startswith("pick-"):
            cb.set_value(cb.key == "pick-pm-prd")
    at.multiselect(key="model-select").set_value(["m1", "m2"])
    at.run()
    at.selectbox(key="baseline-select").set_value("m1")
    at.button(key="generate").click()
    at.run()


def test_setup_explains_the_flow(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    md = _md(at)
    assert "Which model do you actually prefer?" in md
    assert "Rate blind" in md and "Reveal" in md
    assert at.radio(key="bench-step").value == "setup"
    assert sum(1 for c in at.checkbox if c.key and c.key.startswith("pick-")) == 7


def test_blind_session_end_to_end(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    _start_one_task_session(at)
    assert not at.exception
    # generating moves you straight to rating
    assert at.radio(key="bench-step").value == "rate"
    store = BenchStore(env / "r.db")
    [session] = store.sessions()
    outputs = store.outputs(session.session_id)
    md = _md(at)
    assert "Response A" in md and "Response B" in md
    assert "m1" not in [b.label for b in at.button]  # identities stay hidden

    for o in outputs:
        at.radio(key=f"score-{o.id}").set_value(4 if o.model_id == "m2" else 2)
        at.text_input(key=f"liked-{o.id}").input(f"liked {o.label}")
        at.button(key=f"save-{o.id}").click()
        at.run()
        assert not at.exception

    at.button(key="reveal").click()
    at.run()
    assert not at.exception
    assert at.radio(key="bench-step").value == "reveal"
    assert "You preferred m2" in _md(at)
    assert store.session(session.session_id).revealed_at


def test_reveal_button_disabled_until_all_rated(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    _start_one_task_session(at)
    assert at.button(key="reveal").disabled


def test_history_step_renders_empty(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    at.radio(key="bench-step").set_value("history")
    at.run()
    assert not at.exception
    assert "No revealed sessions yet" in _md(at)


def test_create_custom_benchmark_from_the_page(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    at.radio(key="bench-step").set_value("create")
    at.run()
    at.text_input(key="new-title").input("Quarterly planning memo")
    at.text_area(key="new-prompt").input("Draft my Q3 planning memo from these notes: ...")
    at.text_area(key="new-guidance").input("Clear priorities, explicit cuts.")
    at.button(key="new-save").click()
    at.run()
    assert not at.exception
    assert (env / "custom" / "quarterly-planning-memo.yaml").exists()


def test_main_app_still_renders_router_page(env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
