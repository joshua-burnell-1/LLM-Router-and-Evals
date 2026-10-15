import json
from types import SimpleNamespace

import pytest

from routewise.models.cache import ResponseCache
from routewise.models.client import (
    ClaudeCodeBackend,
    LiteLLMBackend,
    MockBackend,
    PermanentError,
    TransientError,
    chat,
    is_retryable,
    render_messages,
    retry_async,
)
from routewise.models.registry import ModelSpec

SPEC = ModelSpec(
    id="m",
    provider="mock",
    cli_model="haiku",
    litellm_model="anthropic/m",
    input_price=1.0,
    output_price=10.0,
)
REGISTRY = {"m": SPEC}


class Sleeper:
    def __init__(self):
        self.delays = []

    async def __call__(self, delay):
        self.delays.append(delay)


async def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("overloaded")
        return "ok"

    sleep = Sleeper()
    assert await retry_async(flaky, attempts=4, base_delay=1.0, jitter=0, sleep=sleep) == "ok"
    assert calls["n"] == 3
    assert sleep.delays == [1.0, 2.0]


async def test_retry_gives_up_after_attempts():
    calls = {"n": 0}

    async def always_fails():
        calls["n"] += 1
        raise TransientError(f"fail {calls['n']}")

    with pytest.raises(TransientError, match="fail 3"):
        await retry_async(always_fails, attempts=3, jitter=0, sleep=Sleeper())
    assert calls["n"] == 3


async def test_retry_does_not_retry_permanent_errors():
    calls = {"n": 0}

    async def bad_request():
        calls["n"] += 1
        raise PermanentError("invalid model")

    with pytest.raises(PermanentError):
        await retry_async(bad_request, attempts=5, sleep=Sleeper())
    assert calls["n"] == 1


async def test_retry_delay_is_capped():
    async def always_fails():
        raise TransientError("x")

    sleep = Sleeper()
    with pytest.raises(TransientError):
        await retry_async(
            always_fails, attempts=5, base_delay=4, max_delay=10, jitter=0, sleep=sleep
        )
    assert sleep.delays == [4, 8, 10, 10]


def test_provider_rate_limit_errors_are_retryable():
    RateLimitError = type("RateLimitError", (Exception,), {})
    BadRequestError = type("BadRequestError", (Exception,), {})
    assert is_retryable(RateLimitError("slow down"))
    assert is_retryable(TimeoutError())
    assert not is_retryable(BadRequestError("nope"))


def test_render_single_user_message_is_plain_prompt():
    system, prompt = render_messages([{"role": "user", "content": "hi"}])
    assert (system, prompt) == (None, "hi")


def test_render_multi_turn_is_transcript_with_system_split_out():
    system, prompt = render_messages(
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )
    assert system == "be nice"
    assert prompt == "User: a\n\nAssistant: b\n\nUser: c"


async def test_chat_with_mock_backend_prices_the_call():
    backend = MockBackend(lambda spec, messages: "x" * 400)
    result = await chat(
        "m",
        [{"role": "user", "content": "y" * 4000}],
        registry=REGISTRY,
        backends={"mock": backend},
    )
    assert result.text == "x" * 400
    assert (result.input_tokens, result.output_tokens) == (1000, 100)
    # 1000 * $1/M + 100 * $10/M
    assert result.cost_usd == pytest.approx(0.002)
    assert result.backend == "mock"
    assert result.tokens_estimated


async def test_chat_retries_transient_backend_errors():
    attempts = {"n": 0}

    def responder(spec, messages):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TransientError("blip")
        return "fine"

    result = await chat(
        "m",
        [{"role": "user", "content": "q"}],
        registry=REGISTRY,
        backends={"mock": MockBackend(responder)},
        retry_sleep=Sleeper(),
    )
    assert result.text == "fine"
    assert attempts["n"] == 2


async def test_chat_uses_cache(tmp_path):
    calls = {"n": 0}

    def responder(spec, messages):
        calls["n"] += 1
        return "answer"

    cache = ResponseCache(tmp_path / "cache.db")
    kwargs = dict(registry=REGISTRY, backends={"mock": MockBackend(responder)}, cache=cache)
    first = await chat("m", [{"role": "user", "content": "q"}], **kwargs)
    second = await chat("m", [{"role": "user", "content": "q"}], **kwargs)
    assert calls["n"] == 1
    assert not first.cached and second.cached
    assert second.text == "answer" and second.cost_usd == first.cost_usd


def _cli_json(**overrides):
    payload = {
        "type": "result",
        "is_error": False,
        "result": "51",
        "duration_api_ms": 1500,
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 2500, "output_tokens": 149},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_claude_cli_success_parsing():
    raw = ClaudeCodeBackend.parse(_cli_json(), prompt_text="x" * 40)
    assert raw.text == "51"
    assert raw.output_tokens == 149
    assert raw.input_tokens == 10  # estimated from prompt text, not CLI scaffolding
    assert raw.latency_s == 1.5
    assert raw.tokens_estimated


def test_claude_cli_rate_limit_is_transient():
    with pytest.raises(TransientError):
        ClaudeCodeBackend.parse(_cli_json(is_error=True, result="API Error: 429 rate limit"), "x")


def test_claude_cli_other_errors_are_permanent():
    with pytest.raises(PermanentError, match="usage credits"):
        ClaudeCodeBackend.parse(_cli_json(is_error=True, result="requires usage credits"), "x")


def test_claude_cli_garbage_output_is_transient():
    with pytest.raises(TransientError):
        ClaudeCodeBackend.parse("not json", "x")


def test_claude_cli_args_disable_tools_and_set_model():
    args = ClaudeCodeBackend(executable="claude").build_args(SPEC, system="sys")
    assert args[:2] == ["claude", "-p"]
    assert args[args.index("--model") + 1] == "haiku"
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--system-prompt") + 1] == "sys"
    assert "--no-session-persistence" in args


async def test_litellm_backend_maps_response():
    seen = {}

    async def fake_acompletion(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        )

    backend = LiteLLMBackend(acompletion=fake_acompletion)
    raw = await backend.complete(SPEC, [{"role": "user", "content": "q"}], max_tokens=50)
    assert seen["model"] == "anthropic/m" and seen["max_tokens"] == 50
    assert (raw.text, raw.input_tokens, raw.output_tokens) == ("hello", 12, 3)
    assert not raw.tokens_estimated


def test_claude_cli_session_limit_is_quota_exhausted():
    from routewise.models.client import QuotaExhaustedError

    msg = "You've hit your session limit · resets 7:50am (UTC)"
    with pytest.raises(QuotaExhaustedError):
        ClaudeCodeBackend.parse(_cli_json(is_error=True, result=msg), "x")


async def test_litellm_backend_drops_params_a_model_does_not_support():
    # Reasoning models (e.g. OpenAI GPT-6) reject temperature=0; LiteLLM must drop it.
    seen = {}

    async def fake_acompletion(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    await LiteLLMBackend(acompletion=fake_acompletion).complete(
        SPEC, [{"role": "user", "content": "q"}], max_tokens=5
    )
    assert seen["drop_params"] is True


async def test_litellm_out_of_credits_is_quota_exhausted_not_retried():
    from routewise.models.client import QuotaExhaustedError

    RateLimitError = type("RateLimitError", (Exception,), {})
    calls = {"n": 0}

    async def broke(**kwargs):
        calls["n"] += 1
        raise RateLimitError("OpenAIException - You have no credits remaining. Add credits")

    backend = LiteLLMBackend(acompletion=broke)
    with pytest.raises(QuotaExhaustedError):
        await chat(
            "m",
            [{"role": "user", "content": "q"}],
            registry={"m": SPEC},
            backends={"mock": backend},
            retry_sleep=Sleeper(),
        )
    assert calls["n"] == 1


async def test_litellm_plain_rate_limit_is_still_retried():
    RateLimitError = type("RateLimitError", (Exception,), {})
    calls = {"n": 0}

    async def busy(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("Rate limit reached for requests per min")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    result = await chat(
        "m",
        [{"role": "user", "content": "q"}],
        registry={"m": SPEC},
        backends={"mock": LiteLLMBackend(acompletion=busy)},
        retry_sleep=Sleeper(),
    )
    assert result.text == "ok" and calls["n"] == 2
