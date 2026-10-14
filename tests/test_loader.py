import json
import textwrap

import pytest

from routewise.evals.framework import score_output
from routewise.evals.loader import all_tasks, load_task_pack, load_task_packs, render


def write(path, text):
    path.write_text(textwrap.dedent(text))
    return path


def test_render_substitutes_whole_value_keeping_type():
    assert render("{{n}}", {"n": 42}) == 42
    assert render(["{{xs}}"], {"xs": [1, 2]}) == [[1, 2]]


def test_render_inline_and_leaves_other_braces_alone():
    out = render('Return {"a": {{n}}} for {{name}}', {"n": 3, "name": "Bo"})
    assert out == 'Return {"a": 3} for Bo'


def test_render_missing_variable_raises():
    with pytest.raises(KeyError, match="missing"):
        render("{{missing}}", {})


def test_static_tasks_inherit_pack_defaults(tmp_path):
    pack = write(
        tmp_path / "p.yaml",
        """
        suite: demo
        category: safety
        tags: [pack]
        system_prompt: be safe
        tasks:
          - id: demo-1
            prompt: hello
            scoring:
              - type: contains_all
                params: {strings: [hi]}
            reference_output: hi
    """,
    )
    [task] = load_task_pack(pack)
    assert (task.suite, task.category, task.system_prompt) == ("demo", "safety", "be safe")
    assert task.tags == ("pack",)
    assert task.scoring[0].params == {"strings": ["hi"]}


def test_template_expands_inline_cases_with_typed_values(tmp_path):
    pack = write(
        tmp_path / "p.yaml",
        """
        suite: math
        category: reasoning
        templates:
          - id: add-{{i}}
            prompt: "What is {{a}} + {{b}}?"
            scoring:
              - type: exact_match
                params: {expected: "{{answer}}", extract: number}
            reference_output: "{{answer}}"
            cases:
              - {a: 1, b: 2, answer: 3}
              - {a: 10, b: 5, answer: 15}
    """,
    )
    tasks = load_task_pack(pack)
    assert [t.id for t in tasks] == ["add-0", "add-1"]
    assert tasks[1].prompt == "What is 10 + 5?"
    assert tasks[1].scoring[0].params["expected"] == 15


def test_template_cases_from_jsonl_file(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps({"id": f"c{n}", "word": w}) for n, w in enumerate("xy")))
    pack = write(
        tmp_path / "p.yaml",
        """
        suite: words
        category: reasoning
        templates:
          - id: echo-{{id}}
            prompt: "Say {{word}}"
            scoring: [{type: contains_all, params: {strings: ["{{word}}"]}}]
            cases_file: cases.jsonl
    """,
    )
    assert [t.prompt for t in load_task_pack(pack)] == ["Say x", "Say y"]


def test_duplicate_ids_rejected(tmp_path):
    for name in ("a.yaml", "b.yaml"):
        write(
            tmp_path / name,
            """
            suite: dup
            category: safety
            tasks:
              - {id: same, prompt: p, scoring: [{type: contains_all, params: {strings: [p]}}]}
        """,
        )
    with pytest.raises(ValueError, match="duplicate task id"):
        load_task_packs(tmp_path)


def test_all_tasks_filters(tmp_path):
    write(
        tmp_path / "p.yaml",
        """
        suite: extra
        category: voice
        tags: [noisy]
        tasks:
          - {id: v1, prompt: p, scoring: [{type: contains_all, params: {strings: [p]}}]}
    """,
    )
    everything = all_tasks(packs_dir=tmp_path)
    assert len(everything) == 17
    assert [t.id for t in all_tasks(packs_dir=tmp_path, suites=["extra"])] == ["v1"]
    assert len(all_tasks(packs_dir=tmp_path, categories=["voice"])) == 4
    assert [t.id for t in all_tasks(packs_dir=tmp_path, tags=["noisy"])] == ["v1"]
    assert len(all_tasks(packs_dir=tmp_path, include_builtin=False)) == 1


REPO_PACK_TASKS = load_task_packs()


def test_repo_ships_several_packs():
    suites = {t.suite for t in REPO_PACK_TASKS}
    assert len(REPO_PACK_TASKS) >= 40
    assert len(suites) >= 4
    assert len({t.category for t in REPO_PACK_TASKS}) == 5


@pytest.mark.parametrize("task", REPO_PACK_TASKS, ids=[t.id for t in REPO_PACK_TASKS])
async def test_repo_pack_references_score_perfectly(task):
    assert task.reference_output
    result = await score_output(task, task.reference_output, judge=None)
    assert result.value == pytest.approx(1.0), (task.id, result.reason)
