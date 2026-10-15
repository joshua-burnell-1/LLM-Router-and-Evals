import textwrap

import pytest

from routewise.bench.store import BenchStore
from routewise.cli import main


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(
        textwrap.dedent("""
        models:
          - {id: m1, provider: mock, input_price: 1, output_price: 2}
          - {id: m2, provider: mock, input_price: 1, output_price: 2}
    """)
    )
    monkeypatch.setenv("ROUTEWISE_MODELS_CONFIG", str(models))
    monkeypatch.setenv("ROUTEWISE_DB", str(tmp_path / "r.db"))
    monkeypatch.setenv("ROUTEWISE_CUSTOM_BENCHMARKS", str(tmp_path / "custom"))
    return tmp_path


def test_bench_list_shows_pm_suite(env, capsys):
    assert main(["bench", "list"]) == 0
    out = capsys.readouterr().out
    assert "pm-prd" in out and "pm-core" in out


def test_bench_run_generates_a_blind_session(env, capsys):
    assert (
        main(
            [
                "bench",
                "run",
                "--models",
                "m1,m2",
                "--baseline",
                "m1",
                "--bench",
                "pm-prd,pm-exec-update",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    store = BenchStore(env / "r.db")
    [session] = store.sessions()
    assert session.session_id in out
    assert len(store.outputs(session.session_id)) == 4
    assert all(o.output for o in store.outputs(session.session_id))
    # the CLI never prints which model got which label
    assert "A = m" not in out and "m1 ->" not in out


def test_bench_reveal_refuses_unrated_session(env, capsys):
    main(["bench", "run", "--models", "m1,m2", "--baseline", "m1", "--bench", "pm-prd"])
    sid = BenchStore(env / "r.db").sessions()[0].session_id
    assert main(["bench", "reveal", sid]) == 1
    assert "still need a rating" in capsys.readouterr().err


def test_bench_history_runs_with_no_sessions(env, capsys):
    assert main(["bench", "history"]) == 0
    assert "no revealed sessions" in capsys.readouterr().out.lower()
