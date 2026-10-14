"""SQLite result store: one row per (task, task version, model, sample)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from routewise.evals.framework import EvalTask

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    args TEXT,
    git_sha TEXT,
    total_cost_usd REAL,
    n_calls INTEGER
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    task_id TEXT NOT NULL,
    task_version TEXT NOT NULL,
    suite TEXT,
    category TEXT,
    model_id TEXT NOT NULL,
    sample_idx INTEGER NOT NULL DEFAULT 0,
    quality REAL,
    score_detail TEXT,
    latency_s REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    tokens_estimated INTEGER,
    cost_usd REAL,
    judge_model TEXT,
    judge_cost_usd REAL,
    backend TEXT,
    output TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (task_id, task_version, model_id, sample_idx)
);
CREATE INDEX IF NOT EXISTS results_model ON results (model_id, category);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ResultRow:
    task_id: str
    task_version: str
    model_id: str
    created_at: str
    sample_idx: int = 0
    run_id: str | None = None
    suite: str | None = None
    category: str | None = None
    quality: float | None = None
    score_detail: str | None = None
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_estimated: bool = False
    cost_usd: float | None = None
    judge_model: str | None = None
    judge_cost_usd: float | None = None
    backend: str | None = None
    output: str | None = None
    error: str | None = None


_COLUMNS = [f.name for f in fields(ResultRow)]


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class ResultStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def start_run(self, args: dict[str, Any]) -> str:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._db.execute(
            "INSERT INTO runs (run_id, started_at, args, git_sha) VALUES (?, ?, ?, ?)",
            (run_id, now_iso(), json.dumps(args, default=str), _git_sha()),
        )
        self._db.commit()
        return run_id

    def finish_run(self, run_id: str, total_cost_usd: float, n_calls: int) -> None:
        self._db.execute(
            "UPDATE runs SET finished_at = ?, total_cost_usd = ?, n_calls = ? WHERE run_id = ?",
            (now_iso(), total_cost_usd, n_calls, run_id),
        )
        self._db.commit()

    def has(self, task_id: str, task_version: str, model_id: str, sample_idx: int = 0) -> bool:
        """True when a successful result exists. Errored rows do not count, so they rerun."""
        row = self._db.execute(
            "SELECT 1 FROM results WHERE task_id = ? AND task_version = ? AND model_id = ? "
            "AND sample_idx = ? AND error IS NULL",
            (task_id, task_version, model_id, sample_idx),
        ).fetchone()
        return row is not None

    def record(self, row: ResultRow) -> None:
        """Insert or replace the row for (task, version, model, sample)."""
        values = asdict(row)
        values["tokens_estimated"] = int(bool(values["tokens_estimated"]))
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self._db.execute(
            f"INSERT OR REPLACE INTO results ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            [values[c] for c in _COLUMNS],
        )
        self._db.commit()

    def rows(
        self,
        *,
        models: Iterable[str] | None = None,
        include_errors: bool = False,
        current: Iterable[EvalTask] | None = None,
    ) -> list[ResultRow]:
        """Stored rows, optionally limited to some models and to the tasks' current versions."""
        sql = f"SELECT {', '.join(_COLUMNS)} FROM results"
        where, params = [], []
        if not include_errors:
            where.append("error IS NULL")
        if models is not None:
            models = list(models)
            where.append(f"model_id IN ({', '.join('?' for _ in models)})")
            params.extend(models)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id"
        out = []
        for r in self._db.execute(sql, params).fetchall():
            data = dict(r)
            data["tokens_estimated"] = bool(data["tokens_estimated"])
            out.append(ResultRow(**data))
        if current is not None:
            keep = {(t.id, t.version) for t in current}
            out = [r for r in out if (r.task_id, r.task_version) in keep]
        return out

    def runs(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT * FROM runs ORDER BY started_at")]
