"""Generate data/tasks/hard_generated.yaml: harder tasks whose gold answers are computed.

    uv run python scripts/generate_hard_pack.py

Families (all deterministic from fixed seeds):
- code: "what does this program print?" Gold comes from actually running the snippet.
- date: calendar arithmetic. Gold comes from the datetime module.
- logic: 5-person ordering puzzles. Constraints are added until brute force finds exactly one
  solution, so the gold answer is provably unique.
- write: constrained writing (exact word count, first word, required words).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import itertools
import json
import random
from pathlib import Path
from typing import Any

import yaml

OUT = Path(__file__).resolve().parents[1] / "data" / "tasks" / "hard_generated.yaml"

# --- code tracing -------------------------------------------------------------------

CODE_TEMPLATES = [
    # aliasing vs copying
    """a = [{x}, {y}, {z}]
b = a
b.append(a[0] + a[-1])
c = a[:]
c[0] = {w}
print(sum(a) - c[0])
""",
    # dict accumulation with modulo buckets
    """counts = {{}}
for i in range({n}):
    key = i % {m}
    counts[key] = counts.get(key, 0) + i
print(max(counts.values()) - min(counts.values()))
""",
    # string building and slicing
    """s = "{word}"
out = ""
for i, ch in enumerate(s):
    out += ch.upper() if i % {k} == 0 else ch
print(out[::-1])
""",
    # late-binding closures
    """fs = [lambda x, i=i: x * i for i in range({n})]
gs = [lambda x: x * i for i in range({n})]
print(sum(f({v}) for f in fs) - sum(g({v}) for g in gs))
""",
    # floor division and modulo with negatives
    """vals = [{a}, -{b}, {c}]
print([v // {d} for v in vals], [v % {d} for v in vals])
""",
]

WORDS = ["routing", "latency", "evaluate", "gateway", "baseline", "throughput"]


def run_snippet(code: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    return buf.getvalue().strip()


def code_tasks(rng: random.Random) -> list[dict[str, Any]]:
    tasks = []
    for t_idx, template in enumerate(CODE_TEMPLATES):
        for rep in range(2):
            params = {
                "x": rng.randint(2, 9),
                "y": rng.randint(2, 9),
                "z": rng.randint(2, 9),
                "w": rng.randint(10, 30),
                "n": rng.randint(7, 13),
                "m": rng.randint(3, 5),
                "word": rng.choice(WORDS),
                "k": rng.randint(2, 3),
                "v": rng.randint(2, 5),
                "a": rng.randint(7, 20),
                "b": rng.randint(7, 20),
                "c": rng.randint(7, 20),
                "d": rng.randint(3, 4),
            }
            code = template.format(**params)
            output = run_snippet(code)
            tasks.append(
                {
                    "id": f"hard-code-{t_idx}{rep}",
                    "category": "reasoning",
                    "description": "Trace a short Python program and give its exact output.",
                    "prompt": (
                        "What does this Python program print? Reason briefly if you like, then "
                        "give the final line as 'OUTPUT: <exactly what is printed>'.\n\n"
                        f"```python\n{code}```"
                    ),
                    "scoring": [
                        {
                            "type": "exact_match",
                            "params": {
                                "extract": "regex",
                                "pattern": r"OUTPUT:\s*`*([^`\n]+?)`*\s*$",
                                "expected": output,
                            },
                        }
                    ],
                    "reference_output": f"OUTPUT: {output}",
                }
            )
    return tasks


# --- calendar arithmetic ----------------------------------------------------------------


def date_tasks(rng: random.Random) -> list[dict[str, Any]]:
    tasks = []
    for i in range(4):
        start = dt.date(2027, 1, 1) + dt.timedelta(days=rng.randint(0, 600))
        n = rng.randint(40, 400)
        end = start + dt.timedelta(days=n)
        tasks.append(
            {
                "id": f"hard-date-add{i}",
                "category": "reasoning",
                "description": "Add days to a date across month and leap-year boundaries.",
                "prompt": (
                    f"What date is {n} days after {start.isoformat()} ({start:%A}), and what "
                    "weekday is it? Give brief working, then the final line as "
                    "'ANSWER: YYYY-MM-DD, Weekday'."
                ),
                "scoring": [
                    {
                        "type": "exact_match",
                        "params": {
                            "extract": "regex",
                            "pattern": r"ANSWER:\s*(\d{4}-\d{2}-\d{2})",
                            "expected": end.isoformat(),
                        },
                    },
                    {
                        "type": "exact_match",
                        "params": {
                            "extract": "regex",
                            "pattern": r"ANSWER:.*?\b(Monday|Tuesday|Wednesday|Thursday|Friday"
                            r"|Saturday|Sunday)\b",
                            "expected": f"{end:%A}",
                        },
                    },
                ],
                "reference_output": f"ANSWER: {end.isoformat()}, {end:%A}",
            }
        )
    for i in range(4):
        start = dt.date(2027, 1, 1) + dt.timedelta(days=rng.randint(0, 600))
        end = start + dt.timedelta(days=rng.randint(20, 90))
        weekdays = sum(
            1 for k in range((end - start).days + 1) if (start + dt.timedelta(days=k)).weekday() < 5
        )
        tasks.append(
            {
                "id": f"hard-date-weekdays{i}",
                "category": "reasoning",
                "description": "Count weekdays in a date range, inclusive.",
                "prompt": (
                    f"How many weekdays (Monday to Friday) are there from {start.isoformat()} "
                    f"({start:%A}) to {end.isoformat()} ({end:%A}), counting both ends? "
                    "Ignore holidays. Give brief working, then the final line as "
                    "'ANSWER: <number>'."
                ),
                "scoring": [
                    {
                        "type": "exact_match",
                        "params": {
                            "extract": "regex",
                            "pattern": r"ANSWER:\s*(\d+)",
                            "expected": weekdays,
                        },
                    }
                ],
                "reference_output": f"ANSWER: {weekdays}",
            }
        )
    return tasks


# --- logic puzzles -------------------------------------------------------------------------

NAME_POOL = ["Ava", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Juno"]


def _check(kind: str, args: tuple, pos: dict[str, int]) -> bool:
    if kind == "before":
        return pos[args[0]] < pos[args[1]]
    if kind == "immediately_before":
        return pos[args[1]] - pos[args[0]] == 1
    if kind == "not_at":
        return pos[args[0]] != args[1]
    if kind == "adjacent":
        return abs(pos[args[0]] - pos[args[1]]) == 1
    if kind == "not_adjacent":
        return abs(pos[args[0]] - pos[args[1]]) != 1
    if kind == "at_an_end":
        return pos[args[0]] in (1, 5)
    raise ValueError(kind)


def _text(kind: str, args: tuple) -> str:
    return {
        "before": "{0} is somewhere in front of {1}.",
        "immediately_before": "{0} is directly in front of {1}.",
        "not_at": "{0} is not in position {1}.",
        "adjacent": "{0} and {1} stand next to each other.",
        "not_adjacent": "{0} and {1} do not stand next to each other.",
        "at_an_end": "{0} is at one end of the line (position 1 or 5).",
    }[kind].format(*args)


def count_solutions(names: list[str], constraints: list[list]) -> int:
    total = 0
    for perm in itertools.permutations(names):
        pos = {name: i + 1 for i, name in enumerate(perm)}
        if all(_check(kind, tuple(args), pos) for kind, args in constraints):
            total += 1
    return total


def logic_puzzles() -> list[dict[str, Any]]:
    puzzles = []
    for p in range(6):
        rng = random.Random(1000 + p)
        names = rng.sample(NAME_POOL, 5)
        truth = names[:]
        rng.shuffle(truth)
        pos = {name: i + 1 for i, name in enumerate(truth)}
        constraints: list[list] = []
        while count_solutions(names, constraints) > 1:
            kind = rng.choice(
                [
                    "before",
                    "immediately_before",
                    "not_at",
                    "adjacent",
                    "not_adjacent",
                    "at_an_end",
                    "before",
                    "not_at",
                ]
            )
            a, b = rng.sample(names, 2)
            args = (
                (a, rng.randint(1, 5))
                if kind == "not_at"
                else (a,)
                if kind == "at_an_end"
                else (a, b)
            )
            if not _check(kind, args, pos):
                continue
            before = count_solutions(names, constraints)
            if count_solutions(names, constraints + [[kind, list(args)]]) < before:
                constraints.append([kind, list(args)])
        puzzles.append({"names": names, "constraints": constraints, "truth": truth})
    return puzzles


def logic_tasks() -> list[dict[str, Any]]:
    tasks = []
    for i, puzzle in enumerate(logic_puzzles()):
        lines = "\n".join(
            f"{k + 1}. {_text(kind, tuple(args))}"
            for k, (kind, args) in enumerate(puzzle["constraints"])
        )
        answer = {str(k + 1): name for k, name in enumerate(puzzle["truth"])}
        tasks.append(
            {
                "id": f"hard-logic-{i}",
                "category": "reasoning",
                "description": "Five-person ordering puzzle with a unique solution.",
                "prompt": (
                    f"Five people ({', '.join(sorted(puzzle['names']))}) stand in a line in "
                    "positions 1 to 5, where 1 is the front. Exactly one arrangement satisfies "
                    f"all of these constraints:\n{lines}\n"
                    "Think it through, then end with a JSON object mapping each position to a "
                    'name, like {"1": "...", "2": "...", "3": "...", "4": "...", "5": "..."}.'
                ),
                "scoring": [
                    {
                        "type": "exact_match",
                        "params": {"extract": "json_fields", "partial": False, "expected": answer},
                    }
                ],
                "reference_output": json.dumps(answer),
            }
        )
    return tasks


# --- constrained writing --------------------------------------------------------------------

WRITING = [
    (
        "a lighthouse in winter",
        12,
        "Snow",
        ["beam", "harbor"],
        "Snow drifts against the lighthouse while its beam sweeps the frozen harbor.",
    ),
    (
        "a city bus at dawn",
        9,
        "Before",
        ["engine", "commuters"],
        "Before sunrise, the engine hums for sleepy waiting commuters.",
    ),
    (
        "debugging late at night",
        15,
        "Coffee",
        ["stack trace", "midnight"],
        "Coffee cooled beside the keyboard as she read the stack trace "
        "again past midnight tonight.",
    ),
    (
        "a vegetable garden",
        10,
        "Tomatoes",
        ["trellis", "August"],
        "Tomatoes climb the trellis and blush red by late August.",
    ),
    (
        "a chess tournament",
        11,
        "Silence",
        ["clock", "gambit"],
        "Silence filled the hall until her bold gambit stopped his clock.",
    ),
    (
        "a rainy train station",
        13,
        "Umbrellas",
        ["platform", "delayed"],
        "Umbrellas crowded the platform as the delayed train finally rolled into view slowly.",
    ),
]


def writing_tasks() -> list[dict[str, Any]]:
    tasks = []
    for i, (topic, n, first, words, reference) in enumerate(WRITING):
        quoted = " and ".join(f"'{w}'" for w in words)
        tasks.append(
            {
                "id": f"hard-write-{i}",
                "category": "structured_output",
                "description": "Constrained writing: exact word count, first word, required words.",
                "prompt": (
                    f"Write one sentence about {topic}. It must have exactly {n} words, start "
                    f"with the word '{first}', and include {quoted}. Output only the sentence."
                ),
                "scoring": [
                    {
                        "type": "word_count_cap",
                        "params": {"min_words": n, "max_words": n},
                        "weight": 2.0,
                    },
                    {"type": "regex_match", "params": {"pattern": rf"^\W*{first}\b"}},
                    {"type": "contains_all", "params": {"strings": words}},
                ],
                "reference_output": reference,
            }
        )
    return tasks


def build() -> dict[str, Any]:
    rng = random.Random(42)
    tasks = code_tasks(rng) + date_tasks(rng) + logic_tasks() + writing_tasks()
    return {"suite": "scale-hard", "tags": ["generated", "hard"], "tasks": tasks}


class _BlockDumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockDumper.add_representer(str, _str_presenter)


def main() -> None:
    header = (
        "# GENERATED by scripts/generate_hard_pack.py. Do not edit by hand; rerun the script.\n"
    )
    body = yaml.dump(build(), Dumper=_BlockDumper, sort_keys=False, allow_unicode=True, width=100)
    OUT.write_text(header + body)
    print(f"wrote {OUT} ({len(build()['tasks'])} tasks)")


if __name__ == "__main__":
    main()
