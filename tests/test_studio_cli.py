import textwrap

import pytest

from routewise.cli import main
from routewise.studio.model import StudioBenchmark, save
from routewise.studio.templates import TEMPLATES


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(
        textwrap.dedent("""
        models:
          - {id: m1, name: Model One, provider: mock, input_price: 1, output_price: 2}
          - {id: m2, name: Model Two, provider: mock, input_price: 0.1, output_price: 0.2}
    """)
    )
    monkeypatch.setenv("ROUTEWISE_MODELS_CONFIG", str(models))
    monkeypatch.setenv("ROUTEWISE_DB", str(tmp_path / "r.db"))
    return tmp_path


def _word_problems() -> StudioBenchmark:
    return next(t for t in TEMPLATES if t.name == "Word problems")


def test_list_shows_saved_and_templates(env, capsys):
    save(_word_problems())
    assert main(["studio", "list"]) == 0
    out = capsys.readouterr().out
    assert "word-problems" in out and "Invoice extraction" in out


def test_run_saved_benchmark_prints_results_and_writes_report(env, capsys):
    save(_word_problems())
    report = env / "report.html"
    assert main(["studio", "run", "word-problems", "--models", "m1,m2", "--html", str(report)]) == 0
    out = capsys.readouterr().out
    assert (
        "Model One" in out
        and "Model Two" in out
        and "wins on quality" in out
        or "tie at" in out
        or "Best quality" in out
    )
    assert "<html" in report.read_text()


def test_run_from_a_yaml_file(env, capsys):
    path = save(_word_problems(), env / "elsewhere")
    assert main(["studio", "run", str(path), "--models", "m1"]) == 0


def test_min_score_gates_for_ci(env, capsys):
    save(_word_problems())
    # mock models do not solve word problems, so a high bar must fail
    assert main(["studio", "run", "word-problems", "--models", "m1", "--min-score", "0.99"]) == 1
    assert "below" in capsys.readouterr().err


def test_unknown_benchmark_and_model(env, capsys):
    assert main(["studio", "run", "nope", "--models", "m1"]) == 2
    save(_word_problems())
    assert main(["studio", "run", "word-problems", "--models", "zzz"]) == 2


def test_judge_benchmark_requires_judge(env, capsys):
    support = next(t for t in TEMPLATES if t.name == "Support replies")
    save(support)
    assert main(["studio", "run", "support-replies", "--models", "m1"]) == 2
    assert "--judge" in capsys.readouterr().err
