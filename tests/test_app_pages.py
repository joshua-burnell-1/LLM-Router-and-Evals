import os
import stat
import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASH = Path(__file__).resolve().parents[1] / "routewise" / "dashboard"


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(
        textwrap.dedent("""
        models:
          - {id: m1, name: Model One, provider: mock, input_price: 1, output_price: 2}
          - {id: o1, name: Open One, provider: openai, litellm_model: openai/o1,
             input_price: 2, output_price: 8}
          - {id: g1, name: Gem One, provider: gemini, enabled: false}
    """)
    )
    monkeypatch.setenv("ROUTEWISE_MODELS_CONFIG", str(models))
    monkeypatch.setenv("ROUTEWISE_DB", str(tmp_path / "r.db"))
    return tmp_path


def _md(at):
    return "\n".join(m.value for m in at.markdown)


def test_app_opens_on_the_overview(env):
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=60).run()
    assert not at.exception
    md = _md(at)
    assert "Know which AI model to use, for every task" in md
    assert "Benchmark studio" in md and "Blind benchmarks" in md and "Model router" in md
    assert "Models ready to test" in md


@pytest.mark.parametrize(
    "page", ["studio_page.py", "bench_page.py", "router_page.py", "keys_page.py"]
)
def test_every_page_renders_through_the_app(env, page):
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=60).run()
    at.switch_page(str(DASH / page)).run()
    assert not at.exception


def test_keys_page_shows_status_and_never_the_key(env):
    at = AppTest.from_file(str(DASH / "keys_page.py"), default_timeout=60).run()
    assert not at.exception
    table = at.dataframe[0].value
    status = dict(zip(table["Model"], table["Status"], strict=True))
    assert status == {"Model One": "Ready", "Open One": "Needs a key", "Gem One": "Off"}

    at.text_input(key="key-openai").input("sk-test-abcdefgh1234")
    at.button(key="FormSubmitter:key-form-openai-Use until restart").click()
    at.run()
    assert not at.exception
    assert os.environ["OPENAI_API_KEY"] == "sk-test-abcdefgh1234"
    md = _md(at)
    assert "ends …1234" in md and "abcdefgh" not in md
    table = at.dataframe[0].value
    assert dict(zip(table["Model"], table["Status"], strict=True))["Open One"] == "Ready"
    assert not (env / ".env").exists() or "sk-test" not in (env / ".env").read_text()


def test_keys_page_saves_to_the_env_file(env):
    at = AppTest.from_file(str(DASH / "keys_page.py"), default_timeout=60).run()
    at.text_input(key="key-openai").input("sk-saved-9876")
    at.button(key="FormSubmitter:key-form-openai-Save to .env").click()
    at.run()
    assert not at.exception
    path = Path(os.environ["ROUTEWISE_ENV_FILE"])
    assert "OPENAI_API_KEY=sk-saved-9876" in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
