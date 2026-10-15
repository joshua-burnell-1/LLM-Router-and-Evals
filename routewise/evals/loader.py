"""Load eval tasks from YAML task packs so the eval set scales without writing Python.

A pack (data/tasks/*.yaml) holds pack-level defaults, static `tasks`, and `templates` that
expand into one task per case. Cases come inline or from a JSONL/YAML file next to the pack.
Placeholders use {{name}} so ordinary braces (JSON examples in prompts) are left alone; a value
that is exactly "{{name}}" keeps the case value's type (numbers stay numbers).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from routewise.config import TASKS_DIR, load_yaml
from routewise.evals.framework import EvalTask, ScorerSpec
from routewise.evals.tasks import BUILTIN_TASKS

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_TASK_KEYS = {
    "id",
    "prompt",
    "scoring",
    "system_prompt",
    "reference_output",
    "category",
    "suite",
    "tags",
    "max_tokens",
    "description",
}


def render(value: Any, variables: dict[str, Any]) -> Any:
    """Substitute {{name}} placeholders recursively through strings, lists and dicts."""
    if isinstance(value, str):
        whole = _PLACEHOLDER.fullmatch(value.strip())
        if whole:
            return _lookup(whole.group(1), variables)
        return _PLACEHOLDER.sub(lambda m: str(_lookup(m.group(1), variables)), value)
    if isinstance(value, list):
        return [render(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: render(v, variables) for k, v in value.items()}
    return value


def _lookup(name: str, variables: dict[str, Any]) -> Any:
    if name not in variables:
        raise KeyError(f"template variable {name!r} missing; have {sorted(variables)}")
    return variables[name]


def _build_task(raw: dict[str, Any], defaults: dict[str, Any], source: Path) -> EvalTask:
    unknown = set(raw) - _TASK_KEYS
    if unknown:
        raise ValueError(f"{source}: unknown task keys {sorted(unknown)}")
    merged = {**defaults, **raw}
    scoring = [
        ScorerSpec(
            type=s["type"],
            params=s.get("params", {}),
            weight=float(s.get("weight", 1.0)),
            required=bool(s.get("required", False)),
        )
        for s in merged.get("scoring", [])
    ]
    return EvalTask(
        id=str(merged["id"]),
        category=merged["category"],
        prompt=merged["prompt"],
        scoring=scoring,
        system_prompt=merged.get("system_prompt"),
        reference_output=(
            str(merged["reference_output"]) if merged.get("reference_output") is not None else None
        ),
        suite=merged.get("suite", "custom"),
        tags=tuple(merged.get("tags", ())),
        max_tokens=int(merged.get("max_tokens", 1024)),
        description=merged.get("description", ""),
    )


def _load_cases(template: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    cases = list(template.get("cases", []))
    if "cases_file" in template:
        path = source.parent / template["cases_file"]
        if path.suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            cases.extend(json.loads(line) for line in lines if line.strip())
        else:
            cases.extend(load_yaml(path).get("cases", []))
    return cases


def load_task_pack(path: str | Path) -> list[EvalTask]:
    path = Path(path)
    pack = load_yaml(path)
    defaults = {
        k: pack[k]
        for k in ("suite", "category", "system_prompt", "tags", "max_tokens")
        if k in pack
    }
    defaults.setdefault("suite", path.stem)
    tasks = [_build_task(raw, defaults, path) for raw in pack.get("tasks", [])]
    for template in pack.get("templates", []):
        body = {k: v for k, v in template.items() if k not in ("cases", "cases_file")}
        for i, case in enumerate(_load_cases(template, path)):
            tasks.append(_build_task(render(body, {"i": i, **case}), defaults, path))
    return tasks


def _check_unique(tasks: Iterable[EvalTask]) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise ValueError(f"duplicate task id: {task.id}")
        seen.add(task.id)


def load_task_packs(directory: str | Path = TASKS_DIR) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    for path in sorted(Path(directory).glob("*.y*ml")):
        tasks.extend(load_task_pack(path))
    _check_unique(tasks)
    return tasks


def studio_tasks() -> list[EvalTask]:
    """Tasks from every saved studio benchmark that is complete enough to run."""
    from routewise.studio.model import list_saved

    tasks: list[EvalTask] = []
    for bench in list_saved():
        if not bench.validate():
            tasks.extend(bench.to_tasks())
    return tasks


def all_tasks(
    *,
    include_builtin: bool = True,
    include_studio: bool = True,
    packs_dir: str | Path | None = TASKS_DIR,
    suites: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
) -> list[EvalTask]:
    """Built-in tasks, every pack and your saved studio benchmarks, filtered. Order is stable
    (built-ins first)."""
    tasks = list(BUILTIN_TASKS) if include_builtin else []
    if packs_dir is not None and Path(packs_dir).exists():
        tasks.extend(load_task_packs(packs_dir))
    if include_studio:
        tasks.extend(studio_tasks())
    _check_unique(tasks)
    if suites:
        tasks = [t for t in tasks if t.suite in set(suites)]
    if categories:
        tasks = [t for t in tasks if t.category in set(categories)]
    if tags:
        wanted = set(tags)
        tasks = [t for t in tasks if wanted & set(t.tags)]
    if ids:
        tasks = [t for t in tasks if t.id in set(ids)]
    return tasks
