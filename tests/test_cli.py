import pytest

from routewise.cli import main
from routewise.evals.store import ResultRow, ResultStore
from routewise.evals.tasks import BUILTIN_TASKS
from tests.conftest import SYNTHETIC_REGISTRY_YAML

PROFILE = {"big": (0.95, 0.010, 3.0), "mid": (0.90, 0.002, 1.5), "small": (0.50, 0.0002, 0.8)}


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(SYNTHETIC_REGISTRY_YAML)
    monkeypatch.setenv("ROUTEWISE_MODELS_CONFIG", str(models))
    db = tmp_path / "results.db"
    monkeypatch.setenv("ROUTEWISE_DB", str(db))
    store = ResultStore(db)
    for task in BUILTIN_TASKS[:8]:
        for model, (q, cost, lat) in PROFILE.items():
            store.record(
                ResultRow(
                    task_id=task.id,
                    task_version=task.version,
                    model_id=model,
                    created_at="2026-10-14T00:00:00+00:00",
                    category=task.category,
                    suite=task.suite,
                    quality=q,
                    cost_usd=cost,
                    latency_s=lat,
                    output="x",
                )
            )
    return tmp_path


def test_models_lists_registry(env, capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "big" in out and "mock" in out and "$4.00" in out


def test_report_prints_tables(env, capsys):
    assert main(["report", "--modes", "balanced"]) == 0
    out = capsys.readouterr().out
    assert "Mean quality" in out and "| Model | reasoning" in out
    assert "router" in out and "flagship" in out


def test_report_writes_file(env, capsys):
    target = env / "results.md"
    assert main(["report", "--out", str(target)]) == 0
    assert "Mean quality" in target.read_text()


def test_route_prints_decision(env, capsys):
    assert main(["route", "Calculate how many units remain", "--mode", "balanced"]) == 0
    out = capsys.readouterr().out
    assert "mid" in out and "reasoning" in out


def test_route_execute_calls_model(env, capsys):
    assert main(["route", "Calculate 2+2", "--mode", "quality", "--execute"]) == 0
    out = capsys.readouterr().out
    assert "[big] Calculate 2+2" in out


def test_replay_prints_summary(env, capsys):
    assert main(["replay", "--mode", "cost"]) == 0
    out = capsys.readouterr().out
    assert "classifier accuracy" in out.lower() and "savings" in out.lower()


def test_snapshot_then_gate(env, capsys):
    baseline = env / "baseline.json"
    assert main(["snapshot", "--out", str(baseline)]) == 0
    assert main(["gate", "--baseline", str(baseline)]) == 0
    store = ResultStore(env / "results.db")
    for row in store.rows():
        if row.model_id == "mid":
            row.quality = 0.2
            store.record(row)
    assert main(["gate", "--baseline", str(baseline)]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_eval_delegates_to_runner(env, capsys):
    assert main(["eval", "--dry-run", "--limit", "2"]) == 0
    assert "2 tasks" in capsys.readouterr().out


def test_rescore_reports_count(env, capsys):
    assert main(["rescore"]) == 0
    assert "rescored" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 2


def test_snapshot_and_gate_can_be_scoped_to_a_suite(env, capsys):
    import json

    baseline = env / "core.json"
    assert main(["snapshot", "--suite", "core", "--out", str(baseline)]) == 0
    data = json.loads(baseline.read_text())
    # only core tasks were seeded, so scoping keeps every cell
    assert data["models"]["big"]["overall"]["n"] == 8
    assert main(["snapshot", "--suite", "scale-hard", "--out", str(env / "hard.json")]) == 0
    assert json.loads((env / "hard.json").read_text())["models"] == {}
    assert main(["gate", "--suite", "core", "--baseline", str(baseline)]) == 0


def test_export_then_import_round_trips(env, tmp_path, capsys):
    out = tmp_path / "seed.jsonl"
    assert main(["export", "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 24
    fresh = tmp_path / "fresh.db"
    assert main(["--db", str(fresh), "import", str(out)]) == 0
    imported = ResultStore(fresh).rows()
    original = ResultStore(env / "results.db").rows()
    key = lambda r: (r.task_id, r.model_id, r.sample_idx)  # noqa: E731
    assert sorted(map(key, imported)) == sorted(map(key, original))
    assert "imported 24" in capsys.readouterr().out
