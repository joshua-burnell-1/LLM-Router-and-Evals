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
