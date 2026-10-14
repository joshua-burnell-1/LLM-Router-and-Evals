import pytest

from routewise.evals.framework import EvalTask, ScorerSpec
from routewise.models.client import ChatResult


def make_task(task_id="t1", category="reasoning", prompt="2+2?", expected="4"):
    return EvalTask(
        id=task_id,
        category=category,
        prompt=prompt,
        scoring=[ScorerSpec("exact_match", {"expected": expected, "extract": "number"})],
        reference_output=expected,
    )


class FakeChat:
    """Stand-in for routewise.models.client.chat with scripted replies per model."""

    def __init__(self, replies=None, cost=0.01, latency=0.5, fail=None):
        self.replies = replies or {}
        self.cost = cost
        self.latency = latency
        self.fail = fail or {}
        self.calls = []

    async def __call__(self, model_id, messages, **kwargs):
        self.calls.append((model_id, messages[-1]["content"]))
        if self.fail.get(model_id):
            self.fail[model_id] -= 1
            raise RuntimeError(f"{model_id} exploded")
        text = self.replies.get(model_id, "4")
        return ChatResult(model_id, text, 100, 20, self.latency, self.cost, "mock", True)


@pytest.fixture
def fake_chat():
    return FakeChat()


def synthetic_rows():
    """Three models with a clear quality/cost/latency trade-off in two categories."""
    from routewise.evals.store import ResultRow

    profile = {
        # model: {category: (quality, cost, latency)}
        "big": {"reasoning": (0.95, 0.010, 3.0), "safety": (0.95, 0.010, 3.0)},
        "mid": {"reasoning": (0.90, 0.002, 1.5), "safety": (0.94, 0.002, 1.5)},
        "small": {"reasoning": (0.60, 0.0002, 0.8), "safety": (0.93, 0.0002, 0.8)},
    }
    rows = []
    for model, cats in profile.items():
        for category, (q, cost, lat) in cats.items():
            for i in range(4):
                rows.append(
                    ResultRow(
                        task_id=f"{category}-{i}",
                        task_version="v1",
                        model_id=model,
                        created_at="2026-10-14T00:00:00+00:00",
                        category=category,
                        quality=q,
                        cost_usd=cost,
                        latency_s=lat,
                        input_tokens=100,
                        output_tokens=50,
                    )
                )
    return rows


SYNTHETIC_REGISTRY_YAML = """
models:
  - {id: big, provider: mock, input_price: 4, output_price: 20}
  - {id: mid, provider: mock, input_price: 2, output_price: 10}
  - {id: small, provider: mock, input_price: 0.1, output_price: 0.5}
"""


@pytest.fixture
def synthetic_registry(tmp_path):
    from routewise.models.registry import load_models

    path = tmp_path / "models.yaml"
    path.write_text(SYNTHETIC_REGISTRY_YAML)
    return load_models(path)
