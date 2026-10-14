"""Regression gate: compare the current leaderboard with a saved snapshot.

    routewise snapshot            # save results/baseline.json
    routewise gate                # exit 1 if any (model, category) quality dropped too far

Run it in CI after an eval run to block prompt, model, or config changes that regress quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from routewise.router.leaderboard import Leaderboard


def snapshot(board: Leaderboard) -> dict[str, Any]:
    models: dict[str, dict[str, dict[str, float]]] = {}
    for record in board.records():
        models.setdefault(record["model_id"], {})[record["category"]] = {
            "mean_quality": round(record["mean_quality"], 4),
            "n": record["n"],
        }
    return {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), "models": models}


def save_snapshot(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_snapshot(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


@dataclass
class Regression:
    model_id: str
    category: str
    baseline: float
    current: float

    @property
    def drop(self) -> float:
        return self.baseline - self.current


@dataclass
class GateResult:
    passed: bool
    checked: int
    regressions: list[Regression] = field(default_factory=list)
    missing: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            note = f"; {len(self.missing)} baseline cells missing (allowed)" if self.missing else ""
            return f"gate passed: {self.checked} (model, category) cells within threshold{note}"
        lines = [
            f"gate FAILED: {len(self.regressions)} regression(s), "
            f"{len(self.missing)} baseline cell(s) missing or under-sampled, "
            f"{self.checked} checked"
        ]
        for model, category in self.missing:
            lines.append(f"  missing: {model} / {category}")
        for r in self.regressions:
            lines.append(
                f"  {r.model_id} / {r.category}: {r.baseline:.3f} -> {r.current:.3f} "
                f"(-{r.drop:.3f})"
            )
        return "\n".join(lines)


def gate(
    board: Leaderboard,
    baseline: dict[str, Any],
    *,
    threshold: float = 0.05,
    min_n: int = 1,
    allow_missing: bool = False,
) -> GateResult:
    """Fail on a regression, on a baseline cell with no (or too few) current results, and on
    checking nothing at all. `allow_missing` tolerates missing cells, never an empty check."""
    regressions, missing, checked = [], [], 0
    for model, cats in baseline.get("models", {}).items():
        for category, base in cats.items():
            cell = board.get(model, category)
            if cell is None or cell.n < min_n:
                missing.append((model, category))
                continue
            checked += 1
            reg = Regression(model, category, base["mean_quality"], cell.mean_quality)
            if reg.drop > threshold:
                regressions.append(reg)
    passed = not regressions and checked > 0 and (allow_missing or not missing)
    return GateResult(passed, checked, regressions, missing)
