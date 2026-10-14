"""OpenAI-compatible routing endpoint.

    uv run routewise serve --port 8000

Point any OpenAI client at http://localhost:8000/v1 and use model "routewise/<mode>"
(quality, balanced, cost) to route per request, or a concrete model id to pass through:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    client.chat.completions.create(model="routewise/balanced", messages=[...])

Responses add a `routewise` object (mode, category, rationale, failover attempts, cost) and
x-routewise-model / x-routewise-category headers.
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from routewise.models.client import ChatResult, chat
from routewise.models.registry import ModelSpec, load_models
from routewise.router.router import NoDataError, Router, default_router


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    stream: bool = False


def _completion(result: ChatResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        },
    }
    if extra:
        body["routewise"] = extra
    return body


def create_app(
    router: Router | None = None,
    chat_fn: Callable[..., Awaitable[ChatResult]] | None = None,
    registry: Mapping[str, ModelSpec] | None = None,
) -> FastAPI:
    app = FastAPI(title="routewise", version="0.1.0")
    state: dict[str, Any] = {"router": router}
    registry = registry if registry is not None else load_models(include_disabled=True)
    chat_fn = chat_fn or functools.partial(chat, registry=registry)

    def get_router() -> Router:
        if state["router"] is None:
            state["router"] = default_router()
        return state["router"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        modes = sorted(get_router().config.modes)
        ids = [f"routewise/{m}" for m in modes] + [m for m, s in registry.items() if s.enabled]
        return {
            "object": "list",
            "data": [{"id": i, "object": "model", "owned_by": "routewise"} for i in ids],
        }

    @app.post("/v1/chat/completions")
    async def completions(req: ChatRequest, response: Response) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(400, "streaming is not supported yet; set stream=false")
        messages = [m.model_dump() for m in req.messages]
        max_tokens = req.max_tokens or 1024

        if req.model == "routewise" or req.model.startswith("routewise/"):
            r = get_router()
            mode = req.model.partition("/")[2] or r.config.default_mode
            if mode not in r.config.modes:
                raise HTTPException(
                    400, f"unknown mode {mode!r}; use one of {sorted(r.config.modes)}"
                )
            try:
                decision, result, attempts = await r.complete(
                    messages, mode, chat_fn=chat_fn, max_tokens=max_tokens
                )
            except NoDataError as exc:
                raise HTTPException(503, str(exc)) from exc
            response.headers["x-routewise-model"] = result.model_id
            response.headers["x-routewise-category"] = decision.category
            return _completion(
                result,
                {
                    "mode": mode,
                    "category": decision.category,
                    "routed_model": decision.model_id,
                    "rationale": decision.rationale,
                    "attempts": attempts,
                    "cost_usd": result.cost_usd,
                    "latency_s": result.latency_s,
                },
            )

        if req.model not in registry:
            raise HTTPException(404, f"unknown model {req.model!r}")
        result = await chat_fn(req.model, messages, max_tokens=max_tokens)
        return _completion(result)

    return app


app = create_app()
