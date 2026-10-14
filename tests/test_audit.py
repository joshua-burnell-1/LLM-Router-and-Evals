from routewise.cli import main
from routewise.evals.audit import audit
from routewise.evals.store import ResultRow


def row(task, model, q, sample=0):
    return ResultRow(
        task_id=task,
        task_version="v",
        model_id=model,
        created_at="x",
        category="reasoning",
        quality=q,
        sample_idx=sample,
    )


ROWS = [
    # every model loses the same 0.25: probably the gold answer, not the models
    row("suspect", "a", 0.75),
    row("suspect", "b", 0.75),
    row("suspect", "c", 0.75),
    # everyone perfect: does not separate models
    row("easy", "a", 1.0),
    row("easy", "b", 1.0),
    row("easy", "c", 1.0),
    # models differ: useful for routing
    row("useful", "a", 1.0),
    row("useful", "b", 0.5),
    row("useful", "c", 0.0),
    # model a disagrees with itself across samples
    row("flaky", "a", 1.0, 0),
    row("flaky", "a", 0.0, 1),
    row("flaky", "b", 1.0),
]


def test_audit_buckets_tasks():
    report = audit(ROWS)
    assert [f.task_id for f in report.suspect_gold] == ["suspect"]
    assert set(report.saturated) == {"easy"}
    assert {f.task_id for f in report.discriminative} == {"useful", "flaky"}
    assert [(f.task_id, f.model_id) for f in report.flaky] == [("flaky", "a")]


def test_audit_ignores_tasks_with_one_model():
    report = audit([row("solo", "a", 0.5)])
    assert report.suspect_gold == [] and report.saturated == []


def test_cli_audit_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROUTEWISE_DB", str(tmp_path / "empty.db"))
    assert main(["audit"]) == 0
    assert "suspect gold" in capsys.readouterr().out.lower()
