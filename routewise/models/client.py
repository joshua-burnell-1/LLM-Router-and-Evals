"""One async `chat()` for every provider.

Backends:
- litellm: real provider APIs (Anthropic, OpenAI, Gemini) using keys from the environment.
- claude_code: the local `claude -p` CLI, so Claude models can be evaluated on a Claude plan
  with no API key. Output tokens and API latency come from the CLI; input tokens are estimated
  from the prompt text because the CLI adds its own scaffolding tokens.
- mock: deterministic responses for tests and offline demos.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar

from routewise.models.cache import ResponseCache
from routewise.models.registry import (
    ModelSpec,
    cost_usd,
    get_model,
    load_models,
    resolve_backend,
)

Message = dict[str, str]
T = TypeVar("T")

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class TransientError(RuntimeError):
    """Worth retrying: rate limits, overload, timeouts, flaky transport."""


class PermanentError(RuntimeError):
    """Not worth retrying: bad request, unknown model, no credits."""


@dataclass
class RawCompletion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    tokens_estimated: bool = False


@dataclass
class ChatResult:
    model_id: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    backend: str
    tokens_estimated: bool = False
    cached: bool = False


class Backend(Protocol):
    async def complete(
        self, spec: ModelSpec, messages: list[Message], *, max_tokens: int
    ) -> RawCompletion: ...


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English)."""
    return math.ceil(len(text) / 4)


# Exception class names (from LiteLLM / provider SDKs) that mean "try again later".
_RETRYABLE_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "Timeout",
    "ServiceUnavailableError",
    "InternalServerError",
}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, TransientError | TimeoutError | ConnectionError):
        return True
    return any(cls.__name__ in _RETRYABLE_NAMES for cls in type(exc).__mro__)


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    retryable: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> T:
    """Call `fn` until it succeeds, backing off exponentially on retryable errors."""
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            if attempt == attempts - 1 or not retryable(exc):
                raise
            delay = min(max_delay, base_delay * 2**attempt)
            if jitter:
                delay *= 1 + random.uniform(-jitter, jitter)
            await sleep(delay)
    raise AssertionError("unreachable")


def render_messages(messages: list[Message]) -> tuple[str | None, str]:
    """Split out the system prompt and flatten the rest into one prompt string."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    turns = [m for m in messages if m["role"] != "system"]
    if len(turns) == 1 and turns[0]["role"] == "user":
        return system, turns[0]["content"]
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in turns]
    return system, "\n\n".join(lines)


class LiteLLMBackend:
    def __init__(self, acompletion: Callable[..., Awaitable[Any]] | None = None):
        self._acompletion = acompletion

    async def complete(
        self, spec: ModelSpec, messages: list[Message], *, max_tokens: int
    ) -> RawCompletion:
        acompletion = self._acompletion
        if acompletion is None:
            import litellm

            acompletion = litellm.acompletion
        start = time.perf_counter()
        response = await acompletion(
            model=spec.litellm_model, messages=messages, max_tokens=max_tokens, temperature=0
        )
        latency = time.perf_counter() - start
        usage = response.usage
        return RawCompletion(
            text=response.choices[0].message.content or "",
            input_tokens=int(usage.prompt_tokens or 0),
            output_tokens=int(usage.completion_tokens or 0),
            latency_s=latency,
        )


_TRANSIENT_CLI_MARKERS = ("rate limit", "429", "overloaded", "529", "timeout", "temporarily")


class ClaudeCodeBackend:
    """Runs `claude -p` headlessly with tools disabled, one call per completion."""

    def __init__(self, executable: str | None = None, timeout_s: float = 300.0):
        self.executable = executable or shutil.which("claude") or "claude"
        self.timeout_s = timeout_s
        # An empty working directory keeps any project CLAUDE.md out of the prompt.
        self._cwd = tempfile.mkdtemp(prefix="routewise-cli-")

    def build_args(self, spec: ModelSpec, system: str | None) -> list[str]:
        return [
            self.executable, "-p",
            "--model", spec.cli_model or spec.id,
            "--output-format", "json",
            "--tools", "",
            "--no-session-persistence",
            "--system-prompt", system or DEFAULT_SYSTEM_PROMPT,
        ]

    @staticmethod
    def parse(stdout: str, prompt_text: str) -> RawCompletion:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise TransientError(f"claude CLI returned non-JSON output: {stdout[:200]!r}") from exc
        if payload.get("is_error"):
            message = str(payload.get("result") or payload.get("subtype") or "unknown error")
            if any(marker in message.lower() for marker in _TRANSIENT_CLI_MARKERS):
                raise TransientError(message)
            raise PermanentError(message)
        usage = payload.get("usage") or {}
        return RawCompletion(
            text=payload.get("result") or "",
            input_tokens=estimate_tokens(prompt_text),
            output_tokens=int(usage.get("output_tokens") or 0),
            latency_s=float(payload.get("duration_api_ms") or payload.get("duration_ms") or 0)
            / 1000,
            tokens_estimated=True,
        )

    async def complete(
        self, spec: ModelSpec, messages: list[Message], *, max_tokens: int
    ) -> RawCompletion:
        system, prompt = render_messages(messages)
        proc = await asyncio.create_subprocess_exec(
            *self.build_args(spec, system),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=os.environ.copy(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=self.timeout_s
            )
        except TimeoutError:
            proc.kill()
            raise TransientError(f"claude CLI timed out after {self.timeout_s}s") from None
        if not stdout.strip():
            raise TransientError(f"claude CLI exited {proc.returncode}: {stderr.decode()[:300]}")
        return self.parse(stdout.decode(), (system or DEFAULT_SYSTEM_PROMPT) + prompt)


class MockBackend:
    """Deterministic backend. `responder(spec, messages)` returns text or raises."""

    def __init__(self, responder: Callable[[ModelSpec, list[Message]], str] | None = None):
        self.responder = responder or (lambda spec, messages: f"[{spec.id}] {messages[-1]['content']}")

    async def complete(
        self, spec: ModelSpec, messages: list[Message], *, max_tokens: int
    ) -> RawCompletion:
        start = time.perf_counter()
        text = self.responder(spec, messages)
        prompt = "".join(m["content"] for m in messages)
        return RawCompletion(
            text=text,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
            latency_s=time.perf_counter() - start,
            tokens_estimated=True,
        )


_DEFAULT_BACKENDS: dict[str, Backend] = {}
_DEFAULT_REGISTRY: dict[str, ModelSpec] | None = None


def default_backend(name: str) -> Backend:
    if name not in _DEFAULT_BACKENDS:
        factories: dict[str, Callable[[], Backend]] = {
            "litellm": LiteLLMBackend,
            "claude_code": ClaudeCodeBackend,
            "mock": MockBackend,
        }
        if name not in factories:
            raise ValueError(f"unknown backend {name!r}")
        _DEFAULT_BACKENDS[name] = factories[name]()
    return _DEFAULT_BACKENDS[name]


def default_registry() -> dict[str, ModelSpec]:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = load_models(include_disabled=True)
    return _DEFAULT_REGISTRY


async def chat(
    model_id: str,
    messages: list[Message],
    *,
    max_tokens: int = 1024,
    registry: Mapping[str, ModelSpec] | None = None,
    cache: ResponseCache | None = None,
    backends: Mapping[str, Backend] | None = None,
    retry_attempts: int = 4,
    retry_sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    env: Mapping[str, str] | None = None,
) -> ChatResult:
    """Send `messages` to `model_id` and return text, tokens, latency, and USD cost."""
    spec = get_model(model_id, registry if registry is not None else default_registry())
    backend_name = resolve_backend(spec, env=env)
    key = None
    if cache is not None:
        key = cache.key(spec.id, backend_name, messages, max_tokens)
        hit = cache.get(key)
        if hit is not None:
            return replace(hit, cached=True)
    backend = (backends or {}).get(backend_name) or default_backend(backend_name)
    raw = await retry_async(
        lambda: backend.complete(spec, messages, max_tokens=max_tokens),
        attempts=retry_attempts,
        sleep=retry_sleep,
    )
    result = ChatResult(
        model_id=spec.id,
        text=raw.text,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        latency_s=raw.latency_s,
        cost_usd=cost_usd(spec, raw.input_tokens, raw.output_tokens),
        backend=backend_name,
        tokens_estimated=raw.tokens_estimated,
    )
    if cache is not None and key is not None:
        cache.put(key, result)
    return result
