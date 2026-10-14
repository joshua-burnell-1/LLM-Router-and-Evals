import contextlib
import importlib.util
import io
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_hard_pack.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

PACK = ROOT / "data" / "tasks" / "hard_generated.yaml"


def test_generator_is_deterministic():
    assert gen.build() == gen.build()


def test_committed_pack_matches_generator():
    assert yaml.safe_load(PACK.read_text()) == gen.build()


def test_pack_size_by_family():
    tasks = gen.build()["tasks"]
    families = Counter(t["id"].split("-")[1] for t in tasks)
    assert families == {
        "code": 10,
        "date": 8,
        "logic": 6,
        "write": 6,
        "mult": 4,
        "logic7": 4,
        "trace": 4,
    }


def test_code_trace_gold_matches_real_execution():
    for task in gen.build()["tasks"]:
        if not task["id"].startswith(("hard-code", "hard-trace")):
            continue
        code = re.search(r"```python\n(.*?)```", task["prompt"], re.S).group(1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, {})
        expected = task["scoring"][0]["params"]["expected"]
        assert buf.getvalue().strip() == str(expected), task["id"]


def test_logic_puzzles_have_unique_solutions():
    for puzzle in gen.logic_puzzles():
        assert gen.count_solutions(puzzle["names"], puzzle["constraints"]) == 1


def test_seven_person_puzzles_have_unique_solutions():
    for puzzle in gen.logic_puzzles(n_people=7, count=4, seed_base=2000):
        assert len(puzzle["names"]) == 7
        assert gen.count_solutions(puzzle["names"], puzzle["constraints"]) == 1


def test_original_tasks_unchanged_by_new_families():
    ids = [t["id"] for t in gen.build()["tasks"]]
    assert ids[:30] == [t["id"] for t in yaml.safe_load(PACK.read_text())["tasks"]][:30]


def test_program_output_tasks_compare_exactly():
    for task in gen.build()["tasks"]:
        if task["id"].startswith(("hard-code", "hard-trace")):
            assert task["scoring"][0]["params"]["exact"] is True, task["id"]
