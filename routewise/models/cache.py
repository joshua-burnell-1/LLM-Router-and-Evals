"""On-disk response cache so repeated identical calls (dev loops, judge reruns) cost nothing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from routewise.models.client import ChatResult


class ResponseCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT, created_at REAL)"
        )

    @staticmethod
    def key(model_id: str, backend: str, messages: list[dict[str, Any]], max_tokens: int) -> str:
        blob = json.dumps([model_id, backend, messages, max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> ChatResult | None:
        from routewise.models.client import ChatResult

        row = self._db.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
        return ChatResult(**json.loads(row[0])) if row else None

    def put(self, key: str, result: ChatResult) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
            (key, json.dumps(asdict(result)), time.time()),
        )
        self._db.commit()
