"""A studio benchmark: one prompt template, test cases, and the checks that score answers.

It compiles to ordinary eval tasks (one per test case), so the runner, result store,
leaderboard and router all work on studio benchmarks unchanged.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from routewise.config import DATA_DIR
from routewise.evals.framework import CATEGORIES, EvalTask
from routewise.studio.catalog import CATALOG, build_scorer_spec

_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def studio_dir() -> Path:
    """Where saved studio benchmarks live ($ROUTEWISE_STUDIO_DIR overrides; gitignored)."""
    return Path(os.environ.get("ROUTEWISE_STUDIO_DIR") or DATA_DIR / "studio")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:50] or "benchmark"


@dataclass
class CheckConfig:
    key: str
    values: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    required: bool = False


@dataclass
class StudioBenchmark:
    name: str
    category: str
    prompt: str
    cases: list[dict[str, str]] = field(default_factory=list)
    checks: list[CheckConfig] = field(default_factory=list)
    system_prompt: str | None = None
    description: str = ""
    judge_model: str = ""

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def suite(self) -> str:
        return f"studio-{self.slug}"

    def variables(self) -> list[str]:
        """Placeholders used in the prompt and system prompt, in order of appearance."""
        text = f"{self.system_prompt or ''}\n{self.prompt}"
        return list(dict.fromkeys(_VAR.findall(text)))

    def uses_judge(self) -> bool:
        return any(c.key == "llm_judge" for c in self.checks)

    def effective_cases(self) -> list[dict[str, str]]:
        """Cases to run; a prompt without variables runs once even with no cases."""
        if self.cases:
            return self.cases
        return [] if self.variables() else [{}]

    def validate(self) -> list[str]:
        """Plain-language problems that would stop a run (empty list means ready)."""
        errors: list[str] = []
        if not self.name.strip():
            errors.append("Give the benchmark a name.")
        if not self.prompt.strip():
            errors.append("Write the prompt the models will answer.")
        if self.category not in CATEGORIES:
            errors.append("Pick a task type.")
        if not self.checks:
            errors.append("Add at least one check to score the answers.")
        cases = self.effective_cases()
        if not cases:
            errors.append(
                "Add at least one test case with values for "
                + ", ".join(f"{{{{{v}}}}}" for v in self.variables())
                + "."
            )
        needs_expected = [
            CATALOG[c.key].label
            for c in self.checks
            if c.key in CATALOG and CATALOG[c.key].uses_expected
        ]
        needs_json = any(CATALOG[c.key].expected_json for c in self.checks if c.key in CATALOG)
        for i, case in enumerate(cases, start=1):
            for var in self.variables():
                if not str(case.get(var, "") or "").strip():
                    errors.append(f"Test case {i} has no value for {{{{{var}}}}}.")
            if needs_expected and not str(case.get("expected", "") or "").strip():
                errors.append(
                    f"Test case {i} needs an expected answer (used by “{needs_expected[0]}”)."
                )
            elif needs_json:
                try:
                    if not isinstance(json.loads(case.get("expected", "")), dict):
                        raise ValueError
                except (ValueError, TypeError):
                    errors.append(f"Test case {i}: the expected answer must be a JSON object.")
        for check in self.checks:
            info = CATALOG.get(check.key)
            if info is None:
                errors.append(f"Unknown check {check.key!r}.")
                continue
            try:
                build_scorer_spec(check.key, check.values, check.weight, check.required)
            except ValueError as exc:
                errors.append(f"“{info.label}”: {exc}.")
            if check.key == "keeps_numbers" and check.values.get("column") not in self.variables():
                errors.append(f"“{info.label}”: pick which prompt variable holds the source text.")
        return errors

    def to_tasks(self) -> list[EvalTask]:
        from routewise.evals.loader import render  # local import: loader imports this module

        tasks = []
        for i, case in enumerate(self.effective_cases(), start=1):
            values = {k: ("" if v is None else str(v)) for k, v in case.items()}
            scoring = []
            for check in self.checks:
                spec = build_scorer_spec(check.key, check.values, check.weight, check.required)
                params = render(spec.params, values)
                if CATALOG[check.key].expected_json:
                    params["expected"] = json.loads(params["expected"])
                scoring.append(type(spec)(spec.type, params, spec.weight, spec.required))
            tasks.append(
                EvalTask(
                    id=f"studio-{self.slug}-{i}",
                    category=self.category,
                    prompt=render(self.prompt, values),
                    system_prompt=render(self.system_prompt, values)
                    if self.system_prompt
                    else None,
                    scoring=scoring,
                    suite=self.suite,
                    tags=("studio",),
                    description=self.name,
                )
            )
        return tasks

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudioBenchmark:
        data = dict(data)
        data["checks"] = [CheckConfig(**c) for c in data.get("checks", [])]
        data["cases"] = [
            {str(k): ("" if v is None else str(v)) for k, v in c.items()}
            for c in data.get("cases", [])
        ]
        return cls(**data)


def save(bench: StudioBenchmark, directory: str | Path | None = None) -> Path:
    directory = Path(directory) if directory is not None else studio_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{bench.slug}.yaml"
    path.write_text(yaml.safe_dump(bench.to_dict(), sort_keys=False, allow_unicode=True, width=100))
    return path


def load_saved(directory: str | Path | None = None) -> dict[str, StudioBenchmark]:
    directory = Path(directory) if directory is not None else studio_dir()
    out: dict[str, StudioBenchmark] = {}
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.yaml")):
        bench = StudioBenchmark.from_dict(yaml.safe_load(path.read_text()) or {})
        out[bench.slug] = bench
    return out


def list_saved(directory: str | Path | None = None) -> list[StudioBenchmark]:
    return sorted(load_saved(directory).values(), key=lambda b: b.name.lower())


def delete_saved(slug: str, directory: str | Path | None = None) -> None:
    directory = Path(directory) if directory is not None else studio_dir()
    (directory / f"{slug}.yaml").unlink(missing_ok=True)
