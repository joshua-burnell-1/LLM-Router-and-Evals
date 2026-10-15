"""A studio benchmark: one prompt template, test cases, and the checks that score answers.

It compiles to ordinary eval tasks (one per test case), so the runner, result store,
leaderboard and router all work on studio benchmarks unchanged.
"""

from __future__ import annotations

import hashlib
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
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50].strip("-")
    if slug:
        return slug
    if not text.strip():
        return "benchmark"
    # names with no Latin letters or digits still get distinct, stable ids
    return "benchmark-" + hashlib.sha256(text.strip().encode()).hexdigest()[:8]


# The only check parameters that hold per-test-case placeholders, all put there by the
# catalogue. Everything else (typed words, regexes, rubrics) is used literally.
_TEMPLATED_PARAMS = {
    "expected_answer": ("expected", "strings"),
    "json_fields": ("expected",),
    "llm_judge": ("reference",),
    "keeps_numbers": ("source",),
}


def _needs_expected(check: CheckConfig) -> bool:
    info = CATALOG.get(check.key)
    if info is None:
        return False
    return info.uses_expected or (
        check.key == "llm_judge" and bool(check.values.get("show_expected"))
    )


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
        needs_expected = [CATALOG[c.key].label for c in self.checks if _needs_expected(c)]
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
                    expected = json.loads(case.get("expected", ""))
                    if not isinstance(expected, dict):
                        raise ValueError
                except (ValueError, TypeError):
                    errors.append(f"Test case {i}: the expected answer must be a JSON object.")
                else:
                    if not expected:
                        errors.append(
                            f"Test case {i}: the expected JSON needs at least one field to compare."
                        )
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
        if not errors:
            try:
                self.to_tasks()
            except Exception as exc:  # anything that would break a run is a problem to show
                errors.append(f"The benchmark could not be prepared: {exc}.")
        return errors

    def to_tasks(self) -> list[EvalTask]:
        from routewise.evals.loader import render  # local import: loader imports this module

        tasks = []
        for i, case in enumerate(self.effective_cases(), start=1):
            values = {k: ("" if v is None else str(v)) for k, v in case.items()}
            scoring = []
            for check in self.checks:
                spec = build_scorer_spec(check.key, check.values, check.weight, check.required)
                params = dict(spec.params)
                for key in _TEMPLATED_PARAMS.get(check.key, ()):
                    if key in params:
                        params[key] = render(params[key], values)
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
        """Build from saved data, tolerating unknown keys and loosely typed values."""
        if not isinstance(data, dict):
            raise ValueError("a saved benchmark must be a mapping")

        def text(value: Any) -> str:
            return "" if value is None else str(value)

        checks = []
        for c in data.get("checks") or []:
            if not isinstance(c, dict) or "key" not in c:
                continue
            try:
                weight = float(c.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            values = c.get("values") if isinstance(c.get("values"), dict) else {}
            checks.append(
                CheckConfig(
                    str(c["key"]), values, min(max(weight, 0.5), 5.0), bool(c.get("required"))
                )
            )
        cases = [
            {str(k): text(v) for k, v in c.items()}
            for c in data.get("cases") or []
            if isinstance(c, dict)
        ]
        system = data.get("system_prompt")
        return cls(
            name=text(data.get("name")),
            category=text(data.get("category")),
            prompt=text(data.get("prompt")),
            cases=cases,
            checks=checks,
            system_prompt=text(system) if system else None,
            description=text(data.get("description")),
            judge_model=text(data.get("judge_model")),
        )


def save(bench: StudioBenchmark, directory: str | Path | None = None) -> Path:
    directory = Path(directory) if directory is not None else studio_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{bench.slug}.yaml"
    if path.exists():
        try:
            existing = StudioBenchmark.from_dict(yaml.safe_load(path.read_text()) or {}).name
        except Exception:
            existing = ""
        if existing and existing.strip() != bench.name.strip():
            raise ValueError(
                f"a benchmark called “{existing}” is already saved under this name; "
                "pick a different name, or load that one to edit it"
            )
    path.write_text(yaml.safe_dump(bench.to_dict(), sort_keys=False, allow_unicode=True, width=100))
    return path


def load_saved(
    directory: str | Path | None = None, *, errors: list[str] | None = None
) -> dict[str, StudioBenchmark]:
    """Saved benchmarks by slug. Unreadable files are skipped (and described in `errors`)."""
    directory = Path(directory) if directory is not None else studio_dir()
    out: dict[str, StudioBenchmark] = {}
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.yaml")):
        try:
            bench = StudioBenchmark.from_dict(yaml.safe_load(path.read_text()) or {})
        except Exception as exc:
            if errors is not None:
                errors.append(f"{path.name}: {exc}")
            continue
        out[bench.slug] = bench
    return out


def list_saved(
    directory: str | Path | None = None, *, errors: list[str] | None = None
) -> list[StudioBenchmark]:
    return sorted(load_saved(directory, errors=errors).values(), key=lambda b: b.name.lower())


def delete_saved(slug: str, directory: str | Path | None = None) -> None:
    directory = Path(directory) if directory is not None else studio_dir()
    (directory / f"{slug}.yaml").unlink(missing_ok=True)
