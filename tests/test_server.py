import pytest
from fastapi.testclient import TestClient

from routewise.router.leaderboard import Leaderboard
from routewise.router.router import ModeWeights, Router, RouterConfig
from routewise.server import create_app
from tests.conftest import FakeChat, synthetic_rows

CONFIG = RouterConfig(
    modes={
        "quality": ModeWeights(quality=1.0, cost=0.005, latency=0.005),
        "balanced": ModeWeights(quality=1.0, cost=0.25, latency=0.10, max_quality_drop=0.10),
    },
    min_samples=2,
)


@pytest.fixture
def setup(synthetic_registry):
    chat = FakeChat(replies={"big": "big says 4", "mid": "mid says 4", "small": "small says 4"})
    router = Router(Leaderboard.from_rows(synthetic_rows()), CONFIG, synthetic_registry)
    client = TestClient(create_app(router=router, chat_fn=chat, registry=synthetic_registry))
    return client, chat


def body(model, text="Calculate how many units remain", **extra):
    return {"model": model, "messages": [{"role": "user", "content": text}], **extra}


def test_routed_completion_is_openai_shaped(setup):
    client, chat = setup
    r = client.post("/v1/chat/completions", json=body("routewise/balanced"))
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "mid"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "mid says 4"}
    assert data["usage"]["total_tokens"] == 120
    assert data["routewise"]["category"] == "reasoning"
    assert "mid" in data["routewise"]["rationale"]
    assert r.headers["x-routewise-model"] == "mid"


def test_bare_routewise_uses_default_mode(setup):
    client, _ = setup
    r = client.post("/v1/chat/completions", json=body("routewise"))
    assert r.json()["routewise"]["mode"] == "balanced"


def test_concrete_model_passes_through(setup):
    client, chat = setup
    r = client.post("/v1/chat/completions", json=body("small"))
    assert r.json()["model"] == "small"
    assert chat.calls[-1][0] == "small"
    assert "routewise" not in r.json()


def test_unknown_mode_is_400(setup):
    client, _ = setup
    assert client.post("/v1/chat/completions", json=body("routewise/fastest")).status_code == 400


def test_unknown_model_is_404(setup):
    client, _ = setup
    assert client.post("/v1/chat/completions", json=body("gpt-nope")).status_code == 404


def test_streaming_not_supported(setup):
    client, _ = setup
    r = client.post("/v1/chat/completions", json=body("routewise/balanced", stream=True))
    assert r.status_code == 400


def test_models_endpoint_lists_modes_and_models(setup):
    client, _ = setup
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert {"routewise/quality", "routewise/balanced", "big", "mid", "small"} <= set(ids)


def test_health(setup):
    client, _ = setup
    assert client.get("/health").json() == {"status": "ok"}


def test_empty_messages_rejected(setup):
    client, _ = setup
    r = client.post("/v1/chat/completions", json={"model": "routewise", "messages": []})
    assert r.status_code == 422


def test_non_positive_max_tokens_rejected(setup):
    client, _ = setup
    for bad in (0, -5):
        r = client.post("/v1/chat/completions", json=body("routewise", max_tokens=bad))
        assert r.status_code == 422


def test_content_parts_are_joined(setup):
    client, chat = setup
    payload = {
        "model": "small",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Calculate"},
                    {"type": "text", "text": "how many"},
                ],
            }
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert chat.calls[-1][1] == "Calculate\nhow many"


def test_backend_failure_is_502(synthetic_registry):
    chat = FakeChat(fail={"big": 9, "mid": 9, "small": 9})
    router = Router(Leaderboard.from_rows(synthetic_rows()), CONFIG, synthetic_registry)
    client = TestClient(create_app(router=router, chat_fn=chat, registry=synthetic_registry))
    assert client.post("/v1/chat/completions", json=body("routewise/balanced")).status_code == 502
    assert client.post("/v1/chat/completions", json=body("small")).status_code == 502


def test_disabled_model_is_404(tmp_path):
    from routewise.models.registry import load_models

    path = tmp_path / "m.yaml"
    path.write_text(
        "models:\n  - {id: parked, provider: mock, input_price: 1, output_price: 1, "
        "enabled: false}\n"
    )
    registry = load_models(path, include_disabled=True)
    client = TestClient(create_app(router=None, chat_fn=FakeChat(), registry=registry))
    assert client.post("/v1/chat/completions", json=body("parked")).status_code == 404
