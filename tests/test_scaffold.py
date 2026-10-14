import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "routewise",
        "routewise.models",
        "routewise.evals",
        "routewise.router",
        "routewise.dashboard",
    ],
)
def test_subpackages_import(module):
    importlib.import_module(module)


def test_version_is_exposed():
    import routewise

    assert routewise.__version__ == "0.1.0"


def test_env_example_has_placeholders_only():
    text = (ROOT / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY=" in text
    assert "OPENAI_API_KEY=" in text
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            _, _, value = line.partition("=")
            assert value in ("", "your-key-here"), line


def test_gitignore_covers_secrets_and_results():
    text = (ROOT / ".gitignore").read_text()
    for pattern in (".env", "*.db", "__pycache__/"):
        assert pattern in text
