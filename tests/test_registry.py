import pytest

from routewise.models.registry import (
    MissingCredentialsError,
    ModelSpec,
    UnknownModelError,
    cost_usd,
    flagship,
    get_model,
    load_models,
    resolve_backend,
)

YAML = """
models:
  - id: cheap
    provider: anthropic
    litellm_model: anthropic/cheap
    cli_model: haiku
    input_price: 0.10
    output_price: 0.50
  - id: big
    provider: anthropic
    litellm_model: anthropic/big
    input_price: 4.0
    output_price: 20.0
  - id: gpt
    provider: openai
    litellm_model: openai/gpt
    input_price: 10.0
    output_price: 50.0
    enabled: false
    price_verified: false
"""


@pytest.fixture
def models_file(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(YAML)
    return path


def test_load_models_skips_disabled_by_default(models_file):
    models = load_models(models_file)
    assert list(models) == ["cheap", "big"]
    assert models["cheap"].cli_model == "haiku"


def test_load_models_can_include_disabled(models_file):
    models = load_models(models_file, include_disabled=True)
    assert models["gpt"].enabled is False
    assert models["gpt"].price_verified is False


def test_cost_usd_uses_per_million_prices(models_file):
    spec = load_models(models_file)["big"]
    # 1,000 input at $4/M + 500 output at $20/M = 0.004 + 0.01
    assert cost_usd(spec, 1000, 500) == pytest.approx(0.014)


def test_cost_usd_zero_tokens_is_free(models_file):
    assert cost_usd(load_models(models_file)["cheap"], 0, 0) == 0.0


def test_flagship_is_most_expensive(models_file):
    models = load_models(models_file)
    assert flagship(models.values()).id == "big"


def test_flagship_of_nothing_raises():
    with pytest.raises(ValueError):
        flagship([])


def test_get_model_unknown_raises(models_file):
    with pytest.raises(UnknownModelError):
        get_model("nope", load_models(models_file))


def _spec(**kw):
    base = dict(id="m", provider="anthropic", litellm_model="anthropic/m", cli_model="haiku")
    base.update(kw)
    return ModelSpec(**base)


def test_backend_explicit_wins():
    assert resolve_backend(_spec(backend="mock"), env={}, which=lambda _: None) == "mock"


def test_backend_env_override():
    env = {"ROUTEWISE_BACKEND": "claude_code", "ANTHROPIC_API_KEY": "x"}
    assert resolve_backend(_spec(), env=env, which=lambda _: None) == "claude_code"


def test_backend_auto_prefers_api_key():
    env = {"ANTHROPIC_API_KEY": "x"}
    assert resolve_backend(_spec(), env=env, which=lambda _: "/bin/claude") == "litellm"


def test_backend_auto_falls_back_to_claude_cli():
    assert resolve_backend(_spec(), env={}, which=lambda _: "/bin/claude") == "claude_code"


def test_backend_auto_without_key_or_cli_raises():
    spec = _spec(provider="openai", litellm_model="openai/m", cli_model=None)
    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        resolve_backend(spec, env={}, which=lambda _: "/bin/claude")


def test_mock_provider_resolves_to_mock():
    assert resolve_backend(_spec(provider="mock"), env={}, which=lambda _: None) == "mock"


def test_repo_models_yaml_loads_and_prices_are_positive():
    models = load_models(include_disabled=True)
    assert {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"} <= set(models)
    for spec in models.values():
        assert spec.input_price > 0 and spec.output_price > 0, spec.id
