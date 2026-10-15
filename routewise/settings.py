"""API keys and model readiness, for the Models & keys page.

Keys are only ever written to the local .env file (gitignored, owner-only permissions) or kept
in this process's environment. They are never logged or shown in full.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

from dotenv import set_key

from routewise.config import PROJECT_ROOT
from routewise.models.registry import (
    PROVIDER_KEY_ENV,
    PROVIDER_LABELS,
    MissingCredentialsError,
    ModelSpec,
    resolve_backend,
)

KEY_PROVIDERS = {p: env for p, env in PROVIDER_KEY_ENV.items() if env}


def env_file() -> Path:
    return Path(os.environ.get("ROUTEWISE_ENV_FILE") or PROJECT_ROOT / ".env")


def _env_name(provider: str) -> str:
    if provider not in KEY_PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}")
    return KEY_PROVIDERS[provider]


def _clean(key: str) -> str:
    key = (key or "").strip()
    if not key:
        raise ValueError("paste a key first")
    return key


def use_key_for_session(provider: str, key: str) -> None:
    """Use a key until the app restarts, without writing it anywhere."""
    os.environ[_env_name(provider)] = _clean(key)


def save_key(provider: str, key: str) -> Path:
    """Save a key to the local .env file (owner-read/write only) and use it now."""
    name, key = _env_name(provider), _clean(key)
    path = env_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    os.chmod(path, 0o600)
    set_key(str(path), name, key, quote_mode="never")
    os.environ[name] = key
    return path


def key_status(provider: str) -> str | None:
    """'set, ends …1234' or None; never more than the last four characters."""
    value = os.environ.get(_env_name(provider), "").strip()
    return f"set, ends …{value[-4:]}" if value else None


def model_status(
    spec: ModelSpec, which: Callable[[str], str | None] = shutil.which
) -> tuple[str, str]:
    """(state, how): state is 'ready', 'needs key' or 'off'."""
    if not spec.enabled:
        return "off", "Disabled in config/models.yaml"
    try:
        backend = resolve_backend(spec, which=which)
    except MissingCredentialsError:
        return "needs key", f"Add an {PROVIDER_LABELS.get(spec.provider, spec.provider)} API key"
    if backend == "claude_code":
        return "ready", "Claude plan (claude CLI)"
    if backend == "litellm":
        return "ready", f"{PROVIDER_LABELS.get(spec.provider, spec.provider)} API key"
    return "ready", backend
