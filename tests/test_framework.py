import pytest

from routewise.evals.framework import (
    SCORERS,
    EvalTask,
    JudgeError,
    LLMJudge,
    Score,
    ScorerSpec,
    extract_json,
    score_output,
)
from routewise.models.client import ChatResult


def run(name, output, **params):
    score = SCORERS[name](output, **params)
    assert isinstance(score, Score)
    assert 0.0 <= score.value <= 1.0 and score.reason
    return score.value


# --- helpers ---------------------------------------------------------------


def test_extract_json_handles_fences_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```\nDone.') == {"a": [1, 2]}
    assert extract_json('Result: [{"x": 1}] as requested') == [{"x": 1}]


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_score_rejects_out_of_range():
    with pytest.raises(ValueError):
        Score(1.5, "too high")


# --- exact_match -----------------------------------------------------------


def test_exact_match_text_normalizes_case_and_space():
    assert run("exact_match", "  Paris. ", expected="paris") == 1.0
    assert run("exact_match", "London", expected="paris") == 0.0


def test_exact_match_number_takes_last_number():
    assert (
        run("exact_match", "Step 1: 3*4=12. Total: $1,250.00", expected=1250, extract="number")
        == 1.0
    )
    assert run("exact_match", "The answer is 7", expected=8, extract="number") == 0.0
    assert run("exact_match", "no digits", expected=8, extract="number") == 0.0


def test_exact_match_regex_group():
    out = "ACTION: clarify\nRESPONSE: Sorry, which day?"
    assert (
        run("exact_match", out, expected="clarify", extract="regex", pattern=r"ACTION:\s*(\w+)")
        == 1.0
    )
    assert (
        run("exact_match", out, expected="stop", extract="regex", pattern=r"ACTION:\s*(\w+)") == 0.0
    )


def test_exact_match_accepts_list_of_alternatives():
    assert (
        run("exact_match", "Line 7", expected=["7", "8"], extract="regex", pattern=r"(\d+)") == 1.0
    )


def test_exact_match_json_fields_gives_partial_credit():
    out = '{"date": "2026-11-03", "start": "14:00", "duration_minutes": 30, "title": "x"}'
    expected = {"date": "2026-11-03", "start": "14:00", "duration_minutes": 45}
    assert run("exact_match", out, expected=expected, extract="json_fields") == pytest.approx(2 / 3)


# --- schema_valid ----------------------------------------------------------

SCHEMA = {
    "type": "object",
    "required": ["name", "amount"],
    "properties": {"name": {"type": "string"}, "amount": {"type": "number"}},
}


def test_schema_valid_pass_and_fail():
    assert run("schema_valid", '```json\n{"name": "A", "amount": 3.5}\n```', schema=SCHEMA) == 1.0
    assert run("schema_valid", '{"name": "A", "amount": "3.5"}', schema=SCHEMA) == 0.0
    assert run("schema_valid", "not json", schema=SCHEMA) == 0.0


# --- word_count_cap / bullet_count ------------------------------------------


def test_word_count_cap():
    assert run("word_count_cap", "one two three", max_words=3) == 1.0
    assert run("word_count_cap", "one two three four", max_words=3) == 0.0


def test_bullet_count():
    out = "- a\n- b\n* c"
    assert run("bullet_count", out, expected=3) == 1.0
    assert run("bullet_count", out + "\n1. d", expected=3) == 0.0


# --- number_preservation -----------------------------------------------------


def test_number_preservation_from_source():
    source = "Revenue rose 12.5% to $1,200 across 3 regions."
    assert run("number_preservation", "Up 12.5% to 1200 in 3 regions", source=source) == 1.0
    assert run("number_preservation", "Up 12.5% in 3 regions", source=source) == pytest.approx(
        2 / 3
    )


def test_number_preservation_explicit_list():
    assert run("number_preservation", "We saw 40 ms p99", numbers=["40", "99"]) == 1.0


# --- slot_f1 -----------------------------------------------------------------


def test_slot_f1_dict_exact_and_fuzzy():
    gold = {"intent": "book_flight", "slots": {"destination": "Denver", "date": "~friday"}}
    pred = '{"intent": "book_flight", "slots": {"destination": "denver", "date": "this Friday"}}'
    assert run("slot_f1", pred, gold=gold) == 1.0


def test_slot_f1_penalizes_hallucinated_and_missing_slots():
    gold = {"a": "1", "b": "2"}
    # one right, one missing, one extra -> P = 1/2, R = 1/2
    assert run("slot_f1", '{"a": "1", "c": "9"}', gold=gold) == pytest.approx(0.5)


def test_slot_f1_list_of_items():
    gold = [{"owner": "Priya", "task": "~budget"}, {"owner": "Sam", "task": "~deck"}]
    pred = '[{"owner": "priya", "task": "Send the budget"}, {"owner": "Lee", "task": "deck"}]'
    # 1 TP of 2 predicted and 2 gold
    assert run("slot_f1", pred, gold=gold) == pytest.approx(0.5)


def test_slot_f1_unparseable_is_zero():
    assert run("slot_f1", "nope", gold={"a": 1}) == 0.0


# --- safety scorers ----------------------------------------------------------


def test_pii_absent():
    clean = "Contact [EMAIL] or [PHONE] about the invoice."
    assert run("pii_absent", clean, must_contain=["invoice"]) == 1.0
    assert run("pii_absent", "mail jo@example.com", must_contain=[]) == 0.0
    assert run("pii_absent", "call (415) 555-0134 now") == 0.0
    assert run("pii_absent", "call +1 415.555.0134 now") == 0.0
    assert run("pii_absent", "[EMAIL]", must_contain=["invoice"]) == 0.0


def test_forbidden_strings_and_contains():
    assert run("forbidden_strings", "I can't share that.", strings=["ZEBRA-42"]) == 1.0
    assert run("forbidden_strings", "the code is zebra-42", strings=["ZEBRA-42"]) == 0.0
    assert run("contains_all", "alpha beta", strings=["alpha", "gamma"]) == 0.5
    assert run("contains_any", "alpha beta", strings=["gamma", "beta"]) == 1.0


# --- judge and combination -----------------------------------------------------


def _fake_chat(reply, cost=0.001):
    calls = []

    async def fake(model_id, messages, **kwargs):
        calls.append((model_id, messages))
        return ChatResult(model_id, reply, 10, 5, 0.1, cost, "mock")

    return fake, calls


TASK = EvalTask(
    id="t1",
    category="reasoning",
    prompt="What is 2+2?",
    scoring=[
        ScorerSpec("exact_match", {"expected": 4, "extract": "number"}, weight=1.0),
        ScorerSpec("llm_judge", {"rubric": "5: correct and concise\n1: wrong"}, weight=1.0),
    ],
)


async def test_judge_parses_and_normalizes_score():
    fake, calls = _fake_chat('{"score": 4, "reason": "correct, slightly verbose"}')
    judge = LLMJudge(model_id="judge", chat_fn=fake)
    score, cost = await judge.score(TASK, "4", rubric="5: great\n1: bad")
    assert score.value == pytest.approx(0.75)
    assert "verbose" in score.reason
    assert cost == 0.001
    assert calls[0][0] == "judge"
    # the rubric is sent verbatim so the grading is auditable
    assert "5: great" in calls[0][1][-1]["content"]


async def test_judge_unparseable_raises():
    fake, _ = _fake_chat("I think it is good")
    with pytest.raises(JudgeError):
        await LLMJudge(model_id="judge", chat_fn=fake).score(TASK, "4", rubric="r")


async def test_score_output_combines_weighted_components():
    fake, _ = _fake_chat('{"score": 3, "reason": "ok"}')
    result = await score_output(TASK, "The answer is 4", judge=LLMJudge("judge", chat_fn=fake))
    assert result.value == pytest.approx((1.0 + 0.5) / 2)
    assert [c.scorer for c in result.components] == ["exact_match", "llm_judge"]
    assert result.judge_cost_usd == 0.001


async def test_score_output_without_judge_skips_judge_components():
    result = await score_output(TASK, "4", judge=None)
    assert result.value == 1.0
    assert [c.scorer for c in result.components] == ["exact_match"]


async def test_required_component_failure_zeroes_task():
    task = EvalTask(
        id="t2",
        category="reasoning",
        prompt="p",
        scoring=[
            ScorerSpec("exact_match", {"expected": "7", "extract": "number"}, required=True),
            ScorerSpec("contains_all", {"strings": ["because"]}),
        ],
    )
    result = await score_output(task, "Line 3 because of the loop", judge=None)
    assert result.value == 0.0


def test_task_version_changes_with_prompt():
    a = EvalTask(id="x", category="reasoning", prompt="p1", scoring=[])
    b = EvalTask(id="x", category="reasoning", prompt="p2", scoring=[])
    assert a.version != b.version and len(a.version) == 12


def test_task_messages_include_system_prompt():
    t = EvalTask(id="x", category="safety", prompt="hi", system_prompt="sys", scoring=[])
    assert t.messages() == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def test_unknown_category_rejected():
    with pytest.raises(ValueError):
        EvalTask(id="x", category="poetry", prompt="p", scoring=[])


def test_json_fields_compares_list_values_as_sets():
    expected = {"attendees": ["dana@acme.io", "li@acme.io"]}
    out_ok = '{"attendees": ["Li@acme.io", "dana@acme.io"]}'
    out_bad = '{"attendees": ["dana@acme.io"]}'
    assert run("exact_match", out_ok, expected=expected, extract="json_fields") == 1.0
    assert run("exact_match", out_bad, expected=expected, extract="json_fields") == 0.0


def test_json_fields_all_or_nothing_when_partial_disabled():
    expected = {"a": 1, "b": 2}
    assert (
        run(
            "exact_match",
            '{"a": 1, "b": 3}',
            expected=expected,
            extract="json_fields",
            partial=False,
        )
        == 0.0
    )
    assert (
        run(
            "exact_match",
            '{"a": 1, "b": 2}',
            expected=expected,
            extract="json_fields",
            partial=False,
        )
        == 1.0
    )
