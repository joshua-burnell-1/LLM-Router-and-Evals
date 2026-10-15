"""The catalogue of checks a studio user can pick, described for humans.

Each check maps friendly form values onto one of the eval framework's scorers. Values may use
{{column}} placeholders that are filled in per test case (for example {{expected}}).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from routewise.evals.framework import ScorerSpec

DEFAULT_RUBRIC = """5: Fully correct and complete, follows every instruction, nothing invented.
4: Correct with a minor omission or slightly off format.
3: Partly correct, or correct but misses an instruction.
2: Mostly wrong or misses the point, with some useful content.
1: Wrong, invented facts, or ignores the instructions."""


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str  # text | textarea | int | choice | column | bool
    default: Any = None
    help: str = ""
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckInfo:
    key: str
    label: str
    description: str
    scorer: str
    params: tuple[Param, ...] = ()
    uses_expected: bool = False  # needs an "expected" column in the test cases
    expected_json: bool = False  # the expected column must hold a JSON object
    build: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]] = field(default=None)


def _words(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        items = [str(v).strip() for v in value]
    else:
        items = [w.strip() for w in str(value or "").replace("\n", ",").split(",")]
    items = [w for w in items if w]
    if not items:
        raise ValueError("add at least one word or phrase (separate words with commas)")
    return items


def _int(value: Any, name: str, minimum: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number") from None
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _expected_answer(v: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    mode = v.get("mode", "exact")
    if mode == "contains":
        return "contains_all", {"strings": ["{{expected}}"]}
    if mode == "number":
        return "exact_match", {"expected": "{{expected}}", "extract": "number"}
    if mode == "exact":
        return "exact_match", {"expected": "{{expected}}", "extract": "text"}
    raise ValueError(f"unknown match mode {mode!r}")


def _word_limit(v: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = {"max_words": _int(v.get("max_words"), "maximum words", 1)}
    if v.get("min_words"):
        params["min_words"] = _int(v["min_words"], "minimum words")
        if params["min_words"] > params["max_words"]:
            raise ValueError("minimum words cannot exceed the maximum")
    return "word_count_cap", params


def _json_schema(v: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = v.get("schema") or ""
    try:
        schema = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise ValueError(f"the JSON schema is not valid JSON: {exc.msg}") from None
    if not isinstance(schema, dict):
        raise ValueError("the JSON schema must be a JSON object")
    return "schema_valid", {"schema": schema}


def _regex(v: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    import re

    pattern = str(v.get("pattern") or "").strip()
    if not pattern:
        raise ValueError("enter a pattern")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"the pattern is not a valid regular expression: {exc}") from None
    return "regex_match", {"pattern": pattern}


def _judge(v: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    rubric = str(v.get("rubric") or "").strip()
    if "5:" not in rubric or "1:" not in rubric:
        raise ValueError("the rubric needs anchors for at least 5: and 1:")
    params: dict[str, Any] = {"rubric": rubric}
    if v.get("show_expected"):
        params["reference"] = "{{expected}}"
    return "llm_judge", params


CHECKS: list[CheckInfo] = [
    CheckInfo(
        "expected_answer",
        "Matches the expected answer",
        "Compares the answer with the expected value you give for each test case.",
        "exact_match",
        (
            Param(
                "mode",
                "How to compare",
                "choice",
                "exact",
                choices=("exact", "number", "contains"),
                help="exact: same text (ignoring case and spacing). number: the last number in "
                "the answer equals the expected number. contains: the expected text "
                "appears anywhere.",
            ),
        ),
        uses_expected=True,
        build=_expected_answer,
    ),
    CheckInfo(
        "json_fields",
        "Extracted fields match",
        "The answer is JSON and each field equals the expected JSON for the test case "
        "(partial credit per field).",
        "exact_match",
        (),
        uses_expected=True,
        expected_json=True,
        build=lambda v: ("exact_match", {"expected": "{{expected}}", "extract": "json_fields"}),
    ),
    CheckInfo(
        "json_schema",
        "Valid JSON with the right shape",
        "The answer contains JSON that matches a JSON Schema (types, required keys).",
        "schema_valid",
        (Param("schema", "JSON Schema", "textarea", '{"type": "object", "required": ["answer"]}'),),
        build=_json_schema,
    ),
    CheckInfo(
        "keywords_all",
        "Mentions every keyword",
        "Partial credit for each required word or phrase the answer contains.",
        "contains_all",
        (Param("words", "Required words or phrases (comma separated)", "text", ""),),
        build=lambda v: ("contains_all", {"strings": _words(v.get("words"))}),
    ),
    CheckInfo(
        "keywords_any",
        "Mentions at least one keyword",
        "Passes if the answer contains any of the words or phrases.",
        "contains_any",
        (Param("words", "Words or phrases (comma separated)", "text", ""),),
        build=lambda v: ("contains_any", {"strings": _words(v.get("words"))}),
    ),
    CheckInfo(
        "forbidden",
        "Avoids forbidden words",
        "Fails if the answer contains any of these, e.g. promises, competitor names, secrets.",
        "forbidden_strings",
        (Param("words", "Forbidden words or phrases (comma separated)", "text", ""),),
        build=lambda v: ("forbidden_strings", {"strings": _words(v.get("words"))}),
    ),
    CheckInfo(
        "word_limit",
        "Stays within a word limit",
        "Passes if the answer's length is within the limits.",
        "word_count_cap",
        (
            Param("max_words", "Maximum words", "int", 100),
            Param("min_words", "Minimum words (optional)", "int", 0),
        ),
        build=_word_limit,
    ),
    CheckInfo(
        "keeps_numbers",
        "Keeps every number from the source",
        "For summaries: share of the numbers in a source column that also appear in the answer.",
        "number_preservation",
        (
            Param(
                "column",
                "Source column",
                "column",
                None,
                help="The test-case column holding the text being summarized.",
            ),
        ),
        build=lambda v: ("number_preservation", {"source": f"{{{{{v.get('column')}}}}}"}),
    ),
    CheckInfo(
        "regex",
        "Matches a pattern",
        "Passes if a regular expression matches the answer (for strict formats).",
        "regex_match",
        (Param("pattern", "Regular expression", "text", r"^ANSWER:"),),
        build=_regex,
    ),
    CheckInfo(
        "no_pii",
        "Contains no personal data",
        "Fails if an email address or phone number appears in the answer.",
        "pii_absent",
        (),
        build=lambda v: ("pii_absent", {}),
    ),
    CheckInfo(
        "llm_judge",
        "Judged against your rubric",
        "A judge model grades each answer 1 to 5 against the rubric you write. Use it for tone, "
        "completeness and reasoning quality that rules cannot capture.",
        "llm_judge",
        (
            Param("rubric", "Rubric (anchors for 5 down to 1)", "textarea", DEFAULT_RUBRIC),
            Param("show_expected", "Show the judge the expected answer", "bool", False),
        ),
        build=_judge,
    ),
]

CATALOG: dict[str, CheckInfo] = {c.key: c for c in CHECKS}


def defaults(key: str) -> dict[str, Any]:
    return {p.name: p.default for p in CATALOG[key].params}


def build_scorer_spec(
    key: str, values: dict[str, Any], weight: float = 1.0, required: bool = False
) -> ScorerSpec:
    """Turn a catalogue check plus form values into a framework ScorerSpec (validated)."""
    if key not in CATALOG:
        raise ValueError(f"unknown check {key!r}")
    info = CATALOG[key]
    scorer, params = info.build({**defaults(key), **(values or {})})
    if weight <= 0:
        raise ValueError("weight must be positive")
    return ScorerSpec(scorer, params, weight=float(weight), required=bool(required))
