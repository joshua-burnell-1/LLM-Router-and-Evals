from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _runs(job):
    return [step.get("run", "") for step in job["steps"]]


def test_ci_runs_lint_format_and_tests():
    wf = yaml.safe_load(WORKFLOW.read_text())
    runs = " ".join(_runs(wf["jobs"]["test"]))
    for command in ("uv sync --locked", "ruff check", "ruff format --check", "pytest"):
        assert command in runs


def test_eval_gate_is_manual_and_budget_capped():
    wf = yaml.safe_load(WORKFLOW.read_text())
    triggers = wf[True] if True in wf else wf["on"]  # PyYAML parses bare `on` as True
    assert "workflow_dispatch" in triggers
    job = wf["jobs"]["eval-gate"]
    assert job["if"] == "github.event_name == 'workflow_dispatch'"
    runs = " ".join(_runs(job))
    assert "--max-cost" in runs and "routewise gate --suite core" in runs
    assert "results/baseline-core.json" in runs
