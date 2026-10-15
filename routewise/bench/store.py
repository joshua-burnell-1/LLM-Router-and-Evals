"""SQLite storage for blind benchmark sessions, generated outputs, and human ratings."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS bench_sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT,
    created_at TEXT NOT NULL,
    baseline_model TEXT NOT NULL,
    models TEXT NOT NULL,
    benchmark_ids TEXT NOT NULL,
    seed INTEGER NOT NULL,
    revealed_at TEXT
);
CREATE TABLE IF NOT EXISTS bench_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES bench_sessions (session_id),
    benchmark_id TEXT NOT NULL,
    benchmark_version TEXT NOT NULL,
    model_id TEXT NOT NULL,
    label TEXT NOT NULL,
    output TEXT,
    error TEXT,
    latency_s REAL,
    cost_usd REAL,
    created_at TEXT,
    score INTEGER,
    liked TEXT,
    disliked TEXT,
    rated_at TEXT,
    UNIQUE (session_id, benchmark_id, model_id)
);
"""


@dataclass
class Session:
    session_id: str
    name: str | None
    created_at: str
    baseline_model: str
    models: list[str]
    benchmark_ids: list[str]
    seed: int
    revealed_at: str | None


@dataclass
class Output:
    id: int
    session_id: str
    benchmark_id: str
    benchmark_version: str
    model_id: str
    label: str
    output: str | None
    error: str | None
    latency_s: float | None
    cost_usd: float | None
    created_at: str | None
    score: int | None
    liked: str | None
    disliked: str | None
    rated_at: str | None


class BenchStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def add_session(self, session: Session, outputs: list[tuple[str, str, str, str]]) -> None:
        """`outputs`: (benchmark_id, benchmark_version, model_id, label) placeholders."""
        self._db.execute(
            "INSERT INTO bench_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session.session_id,
                session.name,
                session.created_at,
                session.baseline_model,
                json.dumps(session.models),
                json.dumps(session.benchmark_ids),
                session.seed,
                session.revealed_at,
            ),
        )
        self._db.executemany(
            "INSERT INTO bench_outputs (session_id, benchmark_id, benchmark_version, model_id, "
            "label) VALUES (?, ?, ?, ?, ?)",
            [(session.session_id, *row) for row in outputs],
        )
        self._db.commit()

    @staticmethod
    def _session(row: sqlite3.Row) -> Session:
        data = dict(row)
        data["models"] = json.loads(data["models"])
        data["benchmark_ids"] = json.loads(data["benchmark_ids"])
        return Session(**data)

    def session(self, session_id: str) -> Session:
        row = self._db.execute(
            "SELECT * FROM bench_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no session {session_id!r}")
        return self._session(row)

    def sessions(self) -> list[Session]:
        rows = self._db.execute("SELECT * FROM bench_sessions ORDER BY rowid").fetchall()
        return [self._session(r) for r in rows]

    def outputs(self, session_id: str) -> list[Output]:
        rows = self._db.execute(
            "SELECT * FROM bench_outputs WHERE session_id = ? ORDER BY benchmark_id, label",
            (session_id,),
        ).fetchall()
        return [Output(**dict(r)) for r in rows]

    def output(self, output_id: int) -> Output:
        row = self._db.execute("SELECT * FROM bench_outputs WHERE id = ?", (output_id,)).fetchone()
        if row is None:
            raise KeyError(f"no output {output_id}")
        return Output(**dict(row))

    def set_output(
        self,
        output_id: int,
        *,
        output: str | None,
        error: str | None,
        latency_s: float | None,
        cost_usd: float | None,
        created_at: str,
    ) -> None:
        self._db.execute(
            "UPDATE bench_outputs SET output = ?, error = ?, latency_s = ?, cost_usd = ?, "
            "created_at = ? WHERE id = ?",
            (output, error, latency_s, cost_usd, created_at, output_id),
        )
        self._db.commit()

    def set_rating(
        self, output_id: int, score: int, liked: str, disliked: str, rated_at: str
    ) -> None:
        self._db.execute(
            "UPDATE bench_outputs SET score = ?, liked = ?, disliked = ?, rated_at = ? "
            "WHERE id = ?",
            (score, liked, disliked, rated_at, output_id),
        )
        self._db.commit()

    def mark_revealed(self, session_id: str, when: str) -> None:
        self._db.execute(
            "UPDATE bench_sessions SET revealed_at = COALESCE(revealed_at, ?) WHERE session_id = ?",
            (when, session_id),
        )
        self._db.commit()
