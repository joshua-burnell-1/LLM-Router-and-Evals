import os

from routewise.models.registry import ModelSpec
from routewise.settings import env_file, key_status, model_status, save_key, use_key_for_session


def test_save_key_writes_private_env_file_and_sets_session(tmp_path, monkeypatch):
    target = tmp_path / ".env"
    monkeypatch.setenv("ROUTEWISE_ENV_FILE", str(target))
    save_key("openai", "sk-test-1234567890abcd")
    assert env_file() == target
    assert (
        "OPENAI_API_KEY='sk-test-1234567890abcd'" in target.read_text()
        or "OPENAI_API_KEY=sk-test-1234567890abcd" in target.read_text()
    )
    assert oct(target.stat().st_mode)[-3:] == "600"
    assert os.environ["OPENAI_API_KEY"] == "sk-test-1234567890abcd"
    # saving again replaces rather than duplicates
    save_key("openai", "sk-test-NEW-0000000000")
    assert target.read_text().count("OPENAI_API_KEY") == 1


def test_key_status_is_masked(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijklmnop1234")
    status = key_status("openai")
    assert status == "set, ends …1234"
    assert "abcdefgh" not in status
    monkeypatch.delenv("OPENAI_API_KEY")
    assert key_status("openai") is None


def test_use_key_for_session_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTEWISE_ENV_FILE", str(tmp_path / ".env"))
    use_key_for_session("gemini", "AIza-session-key-9999")
    assert os.environ["GEMINI_API_KEY"] == "AIza-session-key-9999"
    assert not (tmp_path / ".env").exists()


def test_rejects_empty_or_unknown(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("ROUTEWISE_ENV_FILE", str(tmp_path / ".env"))
    with pytest.raises(ValueError):
        save_key("openai", "   ")
    with pytest.raises(ValueError):
        save_key("acme", "k")


def test_model_status(monkeypatch):
    spec = ModelSpec(id="g", provider="openai", litellm_model="openai/g")
    assert model_status(spec, which=lambda _: None) == ("needs key", "Add an OpenAI API key")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert model_status(spec, which=lambda _: None) == ("ready", "OpenAI API key")
    claude = ModelSpec(id="c", provider="anthropic", litellm_model="anthropic/c", cli_model="haiku")
    status = model_status(claude, which=lambda _: "/bin/claude")
    assert status == ("ready", "Claude plan (claude CLI)")
    off = ModelSpec(id="o", provider="gemini", enabled=False)
    assert model_status(off, which=lambda _: None)[0] == "off"
