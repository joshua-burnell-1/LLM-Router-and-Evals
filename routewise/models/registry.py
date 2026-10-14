"""Model registry: which models exist, what they cost, and how to reach them."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from routewise.config import config_path, load_yaml

# Environment variable holding each provider's API key.
PROVIDER_KEY_ENV: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mock": None,
}

BACKENDS = ("litellm", "claude_code", "mock")


class UnknownModelError(KeyError):
    pass


class MissingCredentialsError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    litellm_model: str | None = None
    cli_model: str | None = None
    input_price: float = 0.0  # USD per 1M input tokens
    output_price: float = 0.0  # USD per 1M output tokens
    backend: str = "auto"
    enabled: bool = True
    price_verified: bool = True
    tier: str = ""
    notes: str = ""


def load_models(
    path: str | Path | None = None, *, include_disabled: bool = False
) -> dict[str, ModelSpec]:
    """Load config/models.yaml (or `path`) into an ordered {id: ModelSpec} dict."""
    data = load_yaml(path or config_path("models.yaml"))
    known = {f.name for f in fields(ModelSpec)}
    models: dict[str, ModelSpec] = {}
    for entry in data.get("models", []):
        unknown = set(entry) - known
        if unknown:
            raise ValueError(f"unknown keys for model {entry.get('id')}: {sorted(unknown)}")
        if not isinstance(entry.get("id"), str):
            raise ValueError(
                f"model id {entry.get('id')!r} must be a string; quote it in YAML "
                "(bare on/off/yes/no parse as booleans)"
            )
        spec = ModelSpec(**entry)
        if spec.id in models:
            raise ValueError(f"duplicate model id: {spec.id}")
        if spec.enabled or include_disabled:
            models[spec.id] = spec
    return models


def get_model(model_id: str, registry: Mapping[str, ModelSpec] | None = None) -> ModelSpec:
    registry = registry if registry is not None else load_models(include_disabled=True)
    try:
        return registry[model_id]
    except KeyError:
        raise UnknownModelError(f"unknown model {model_id!r}; known: {sorted(registry)}") from None


def cost_usd(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    """USD cost of one call at the model's list prices."""
    return (input_tokens * spec.input_price + output_tokens * spec.output_price) / 1_000_000


def flagship(specs: Iterable[ModelSpec]) -> ModelSpec:
    """The most expensive model (by output price, then input price): the baseline to beat."""
    specs = list(specs)
    if not specs:
        raise ValueError("no models to pick a flagship from")
    return max(specs, key=lambda s: (s.output_price, s.input_price, s.id))


def resolve_backend(
    spec: ModelSpec,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    """Pick the backend for a model.

    Order: explicit `backend` in models.yaml, then ROUTEWISE_BACKEND, then the provider's API
    key (LiteLLM), then the local `claude` CLI for Anthropic models (uses your Claude plan).
    """
    env = os.environ if env is None else env
    if spec.backend != "auto":
        return spec.backend
    override = env.get("ROUTEWISE_BACKEND", "").strip()
    if override:
        if override not in BACKENDS:
            raise ValueError(f"ROUTEWISE_BACKEND must be one of {BACKENDS}, got {override!r}")
        return override
    if spec.provider == "mock":
        return "mock"
    key_env = PROVIDER_KEY_ENV.get(spec.provider)
    if key_env and env.get(key_env):
        return "litellm"
    if spec.provider == "anthropic" and spec.cli_model and which("claude"):
        return "claude_code"
    raise MissingCredentialsError(
        f"no way to call {spec.id}: set {key_env or 'an API key'}"
        + (" or install the claude CLI" if spec.provider == "anthropic" else "")
    )
