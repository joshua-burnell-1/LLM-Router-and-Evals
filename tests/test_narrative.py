import pytest

from routewise.bench.core import ModelSummary, Reveal
from routewise.dashboard.narrative import (
    category_label,
    detection_note,
    reveal_headline,
    verdict,
)
from routewise.router.classifier import Classification
from routewise.router.router import Candidate

NAMES = {"big": "Big Model", "small": "Small Model", "mid": "Mid Model"}


def cand(model, quality, cost, latency=1.0, eligible=True, n=40):
    return Candidate(model, 0.0, quality, cost, latency, 0, 0, n, "reasoning", eligible)


def test_category_label_is_human_readable():
    assert category_label("structured_output") == "Structured output"
    assert category_label("general") == "General"


def test_verdict_cheaper_at_same_quality():
    head, reason = verdict(
        cand("small", 0.998, 0.0002), cand("big", 0.998, 0.0023), "reasoning", "balanced", NAMES.get
    )
    assert head == "Send this prompt to Small Model"
    assert "Same expected quality as Big Model" in reason
    assert "9% of the cost" in reason and "reasoning" in reason.lower()


def test_verdict_small_quality_tradeoff():
    _, reason = verdict(
        cand("small", 0.95, 0.0002), cand("big", 0.99, 0.0023), "summarization", "cost", NAMES.get
    )
    assert "0.04 lower expected quality than Big Model" in reason


def test_verdict_when_flagship_wins():
    head, reason = verdict(
        cand("big", 0.99, 0.0023), cand("big", 0.99, 0.0023), "voice", "quality", NAMES.get
    )
    assert head == "Send this prompt to Big Model"
    assert "highest expected quality" in reason.lower()


def test_detection_note():
    assert "JSON" in detection_note(Classification("structured_output", 1, "rules", ["JSON"]))
    knn = Classification("voice", 1, "knn", ["asr-1 (0.42)"])
    assert "similar" in detection_note(knn).lower()
    general = Classification("general", 0, "default", [])
    assert "all task types" in detection_note(general)


def summary(model, mean, wins, delta, base=False):
    return ModelSummary(model, mean, 7, wins, delta, base)


def test_reveal_headline_new_model_wins():
    result = Reveal(
        "s", "mid", "big", [summary("big", 4.3, 5, 0.8), summary("mid", 3.5, 2, 0, True)], []
    )
    head, line = reveal_headline(result, NAMES.get, n_tasks=7)
    assert head == "You preferred Big Model"
    assert "4.3 out of 5" in line and "0.8 higher than Mid Model" in line
    assert "your current model" in line and "won 5 of 7 tasks" in line


def test_reveal_headline_current_model_holds():
    result = Reveal(
        "s", "mid", "mid", [summary("mid", 4.1, 4, 0, True), summary("big", 3.9, 3, -0.2)], []
    )
    head, line = reveal_headline(result, NAMES.get, n_tasks=7)
    assert head == "Your current model held its ground"
    assert "Mid Model" in line


def test_reveal_headline_half_wins_formatting():
    result = Reveal(
        "s", "mid", "big", [summary("big", 4.0, 1.5, 0.5), summary("mid", 3.5, 0.5, 0, True)], []
    )
    _, line = reveal_headline(result, NAMES.get, n_tasks=2)
    assert "won 1.5 of 2 tasks" in line


@pytest.mark.parametrize("bad", ["<script>x</script>", "A & B"])
def test_ui_cards_escape_text(bad):
    from routewise.dashboard.ui import model_card

    html = model_card(
        name=bad, provider="p", badge="b", stats=[("Quality", bad, None)], footnote=bad
    )
    assert "<script>" not in html and "&lt;" in html or "&amp;" in html


def test_reveal_headline_reports_a_tie():
    result = Reveal(
        "s",
        "mid",
        "big",
        [
            summary("big", 4.5, 1, 0.0),
            summary("mid", 4.5, 1, 0, True),
            summary("small", 3.5, 0, -1.0),
        ],
        [],
    )
    head, line = reveal_headline(result, NAMES.get, n_tasks=2)
    assert head == "A tie between Big Model and Mid Model"
    assert "4.5 out of 5" in line and "won 1 of 2" in line and "more tasks" in line


def test_tied_top_lists_every_co_leader():
    from routewise.dashboard.narrative import tied_top

    result = Reveal(
        "s",
        "mid",
        "big",
        [
            summary("big", 4.5, 1, 0.0),
            summary("mid", 4.5, 1, 0, True),
            summary("small", 3.5, 0, -1.0),
        ],
        [],
    )
    assert tied_top(result) == ["big", "mid"]
