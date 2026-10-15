import os
import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "routewise" / "dashboard" / "studio_page.py")


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(
        textwrap.dedent("""
        models:
          - {id: m1, name: Model One, provider: mock, input_price: 1, output_price: 2}
          - {id: m2, name: Model Two, provider: mock, input_price: 0.1, output_price: 0.2}
    """)
    )
    monkeypatch.setenv("ROUTEWISE_MODELS_CONFIG", str(models))
    monkeypatch.setenv("ROUTEWISE_DB", str(tmp_path / "r.db"))
    return tmp_path


def _md(at):
    return "\n".join(m.value for m in at.markdown)


def _load(at, option):
    at.selectbox(key="st-start").set_value(option)
    at.button(key="st-load").click()
    at.run()


def test_opens_with_a_ready_template(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    assert "Benchmark any model on your own work" in _md(at)
    assert at.text_input(key="st-name").value == "Invoice extraction"
    assert "Estimated cost" in _md(at)
    assert not at.button(key="st-run").disabled


def test_run_a_template_end_to_end(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    _load(at, "template:3")  # Word problems
    assert at.text_input(key="st-name").value == "Word problems"
    at.multiselect(key="st-models").set_value(["m1", "m2"])
    at.run()
    at.button(key="st-run").click()
    at.run()
    assert not at.exception
    md = _md(at)
    assert "Results" in md and "Score by check" in md
    assert any(t in md for t in ("Best quality", "wins on quality and value", "models tie at"))
    assert "Model One" in md and "Model Two" in md
    assert (Path(os.environ["ROUTEWISE_STUDIO_DIR"]) / "word-problems.yaml").exists()


def test_incomplete_benchmark_explains_and_blocks_the_run(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    _load(at, "template:4")  # Blank benchmark: no test case values
    assert not at.exception
    assert "Before you can run" in _md(at)
    assert at.button(key="st-run").disabled


def test_save_benchmark(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    at.text_input(key="st-name").input("My invoices")
    at.run()
    at.button(key="st-save").click()
    at.run()
    assert not at.exception
    assert (Path(os.environ["ROUTEWISE_STUDIO_DIR"]) / "my-invoices.yaml").exists()
    # it now appears under "Start from"
    assert "Saved · My invoices" in at.selectbox(key="st-start").options


def test_saving_over_a_different_benchmark_is_refused(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    at.text_input(key="st-name").input("My invoices").run()
    at.button(key="st-save").click().run()
    at.text_input(key="st-name").input("My Invoices!").run()
    at.button(key="st-save").click().run()
    assert not at.exception
    assert any("already saved" in e.value for e in at.error)


def test_judge_shown_the_expected_answer_gets_an_expected_column(env):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    _load(at, "template:1")  # Support replies: judged, no expected column
    assert "expected" not in at.session_state["st-cases-cols"]
    at.checkbox(key="chk-llm_judge-show_expected").check().run()
    assert not at.exception
    assert "expected" in at.session_state["st-cases-cols"]
    assert "Before you can run" in _md(at)  # the new column is empty


def test_unreadable_saved_files_are_reported_not_fatal(env):
    d = Path(os.environ["ROUTEWISE_STUDIO_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "broken.yaml").write_text("name: [unclosed")
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    assert any("broken.yaml" in w.value for w in at.warning)
