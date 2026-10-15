"""routewise command line.

routewise models                     list models, prices, and which backend each uses
routewise eval [--limit N ...]       run evals (see `routewise eval --help`)
routewise report [--out FILE]        results, per-category, and router replay tables
routewise route "query" [--mode M]   show the routing decision (--execute to call it)
routewise replay [--mode M]          offline router evaluation vs always-flagship
routewise snapshot / gate            save a quality baseline / fail on regressions
routewise rescore                    rerun scorers on stored outputs (no model calls)
routewise dashboard                  launch the Streamlit dashboard
routewise serve                      OpenAI-compatible routing endpoint
routewise bench list|run|reveal|history   blind, human-rated model comparisons
routewise studio list|run            run a studio benchmark (CI-friendly with --min-score)
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from dotenv import load_dotenv

from routewise.config import PROJECT_ROOT, default_db_path
from routewise.evals.loader import all_tasks
from routewise.evals.store import ResultStore
from routewise.models.registry import MissingCredentialsError, load_models, resolve_backend
from routewise.router.classifier import ExampleIndex
from routewise.router.leaderboard import Leaderboard
from routewise.router.router import Router, load_router_config

DEFAULT_BASELINE = PROJECT_ROOT / "results" / "baseline.json"


class Context:
    """Lazily loaded store, tasks, registry, leaderboard, and router."""

    def __init__(self, db: str | None, suites: list[str] | None = None):
        self.store = ResultStore(db or default_db_path())
        self.tasks = all_tasks(suites=suites)
        self.registry = load_models(include_disabled=True)
        self.config = load_router_config()
        self.board = Leaderboard.from_store(self.store, current=self.tasks)
        self.router = Router(
            self.board, self.config, self.registry, examples=ExampleIndex.from_tasks(self.tasks)
        )


def _suites(args: argparse.Namespace) -> list[str] | None:
    value = getattr(args, "suite", None)
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def cmd_models(args: argparse.Namespace) -> int:
    print("| Model | Provider | Backend | $ in / 1M | $ out / 1M | Price verified | Enabled |")
    print("|---|---|---|---|---|---|---|")
    for spec in load_models(include_disabled=True).values():
        try:
            backend = resolve_backend(spec)
        except MissingCredentialsError:
            backend = "unavailable (no key)"
        print(
            f"| {spec.id} | {spec.provider} | {backend} | ${spec.input_price:.2f} | "
            f"${spec.output_price:.2f} | {'yes' if spec.price_verified else 'NO'} | "
            f"{'yes' if spec.enabled else 'no'} |"
        )
    return 0


def build_report(ctx: Context, modes: Iterable[str]) -> str:
    from routewise.router.replay import replay
    from routewise.router.report import category_table, replay_table, results_table

    summaries = [
        replay(ctx.store.rows(current=ctx.tasks), ctx.tasks, mode, ctx.config, ctx.registry)
        for mode in modes
    ]
    parts = [
        "### Per-model results\n",
        results_table(ctx.board),
        "\n### Mean quality by category\n",
        category_table(ctx.board),
        "\n### Router replay (leave-one-out) vs always-flagship\n",
        replay_table(summaries),
    ]
    for s in summaries:
        parts.append(
            f"\n- **{s.mode}**: {s.n_tasks} tasks, cost savings vs flagship "
            f"{s.savings_vs_flagship:.0%}, quality delta {s.quality_delta_vs_flagship:+.3f}, "
            f"classifier accuracy {s.classifier_accuracy:.0%}"
        )
    return "\n".join(parts) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    ctx = Context(args.db, _suites(args))
    if not ctx.board.models():
        print("no results yet: run `routewise eval --limit 4` first", file=sys.stderr)
        return 2
    text = build_report(ctx, args.modes.split(","))
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    ctx = Context(args.db)
    decision = ctx.router.route(args.query, args.mode)
    baseline = ctx.router.always_flagship(args.query)
    print(f"routed to: {decision.model_id}  (mode {decision.mode}, category {decision.category})")
    print(decision.rationale)
    print(f"flagship baseline: {baseline.model_id}")
    print("\nranking:")
    for c in decision.candidates:
        flag = "" if c.eligible else f"  [excluded: {c.excluded_reason}]"
        print(
            f"  {c.model_id:<22} score {c.score:+.3f}  quality {c.quality:.2f}  "
            f"${c.mean_cost_usd:.5f}  {c.mean_latency_s:.1f}s{flag}"
        )
    if args.execute:
        import functools

        from routewise.models.client import chat

        chat_fn = functools.partial(chat, registry=ctx.registry)
        _, result, attempts = asyncio.run(
            ctx.router.complete(args.query, args.mode, chat_fn=chat_fn)
        )
        print(
            f"\n--- {result.model_id} ({result.latency_s:.1f}s, ${result.cost_usd:.5f}"
            + (f", after failover from {attempts[:-1]}" if len(attempts) > 1 else "")
            + ") ---"
        )
        print(result.text)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from routewise.router.replay import replay
    from routewise.router.report import replay_table

    ctx = Context(args.db)
    s = replay(ctx.store.rows(current=ctx.tasks), ctx.tasks, args.mode, ctx.config, ctx.registry)
    print(replay_table([s]))
    print(
        f"\n{s.n_tasks} tasks | savings vs flagship {s.savings_vs_flagship:.0%} | quality delta "
        f"{s.quality_delta_vs_flagship:+.3f} | classifier accuracy {s.classifier_accuracy:.0%}"
    )
    misses = [t for t in s.per_task if t["predicted_category"] != t["category"]]
    if misses:
        print("\nclassifier misses:")
        for t in misses:
            print(f"  {t['task_id']}: {t['category']} -> {t['predicted_category']}")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    from routewise.router.gate import save_snapshot, snapshot

    ctx = Context(args.db, _suites(args))
    save_snapshot(snapshot(ctx.board), args.out)
    print(f"wrote baseline {args.out}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from routewise.router.gate import gate, load_snapshot

    ctx = Context(args.db, _suites(args))
    result = gate(
        ctx.board,
        load_snapshot(args.baseline),
        threshold=args.threshold,
        min_n=args.min_n,
        allow_missing=args.allow_missing,
    )
    print(result.summary())
    return 0 if result.passed else 1


def cmd_rescore(args: argparse.Namespace) -> int:
    from routewise.evals.rescore import rescore

    ctx = Context(args.db)
    changed = asyncio.run(rescore(ctx.store, ctx.tasks))
    print(f"rescored {changed} rows")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from routewise.evals.audit import audit

    ctx = Context(args.db, _suites(args))
    print(audit(ctx.store.rows(current=ctx.tasks)).summary())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import json
    from dataclasses import asdict

    ctx = Context(args.db)
    rows = ctx.store.rows(current=ctx.tasks)  # current task versions only, no error rows
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} rows to {args.out}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    import json

    from routewise.evals.store import ResultRow

    store = ResultStore(args.db or default_db_path())
    n = 0
    with open(args.path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                store.record(ResultRow(**json.loads(line)))
                n += 1
    print(f"imported {n} rows into {store.path}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    app = Path(__file__).parent / "dashboard" / "app.py"
    # Run from the project root so .streamlit/config.toml (the theme) is picked up.
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)], cwd=PROJECT_ROOT)


def cmd_bench(args: argparse.Namespace) -> int:
    """Blind benchmarks: list, run (generate blind responses), reveal, history."""
    import functools

    from routewise.bench.core import (
        NotCompleteError,
        create_session,
        generate_outputs,
        history,
        load_benchmarks,
        reveal,
    )
    from routewise.bench.store import BenchStore
    from routewise.models.client import chat

    benches = load_benchmarks()
    store = BenchStore(args.db or default_db_path())

    if args.bench_command == "list":
        print("| Benchmark | Suite | Title | Version |")
        print("|---|---|---|---|")
        for b in benches.values():
            print(f"| {b.id} | {b.suite} | {b.title} | {b.version} |")
        return 0

    if args.bench_command == "run":
        registry = load_models(include_disabled=True)
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in models if m not in registry]
        if unknown:
            print(f"unknown models: {unknown}", file=sys.stderr)
            return 2
        if args.bench:
            ids = [b.strip() for b in args.bench.split(",") if b.strip()]
            missing = [b for b in ids if b not in benches]
            if missing:
                print(f"unknown benchmarks: {missing}", file=sys.stderr)
                return 2
            chosen = [benches[b] for b in ids]
        else:
            chosen = [b for b in benches.values() if b.suite == args.suite]
        try:
            sid = create_session(store, chosen, models, baseline=args.baseline, name=args.name)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        chat_fn = functools.partial(chat, registry=registry)
        n = len(chosen) * len(models)
        print(f"session {sid}: generating {n} blind responses...")
        done = asyncio.run(generate_outputs(store, sid, benches, chat_fn=chat_fn, progress=print))
        failed = n - done
        print(f"{done} responses ready" + (f", {failed} failed (rerun to retry)" if failed else ""))
        print("Rate them in the dashboard: `routewise dashboard`, page 'Blind benchmarks'.")
        return 1 if failed else 0

    if args.bench_command == "reveal":
        try:
            result = reveal(store, args.session)
        except NotCompleteError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"You preferred {result.preferred} (baseline: {result.baseline})")
        for m in result.models:
            print(
                f"  {m.model_id:<22} mean {m.mean_score:.2f}  wins {m.wins:g}  "
                f"vs baseline {m.delta_vs_baseline:+.2f}"
            )
        return 0

    if args.bench_command == "history":
        rows = history(store, benches)
        if not rows:
            print("No revealed sessions yet.")
            return 0
        print("| Model | Mean human score | Ratings | Sessions | First | Last |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            print(
                f"| {r.model_id} | {r.mean_score:.2f} | {r.ratings} | {r.sessions} | "
                f"{r.first_at[:10]} | {r.last_at[:10]} |"
            )
        return 0
    return 2


def cmd_studio(args: argparse.Namespace) -> int:
    """Studio benchmarks from the command line: list them, or run one and print the results."""
    import functools

    import yaml

    from routewise.evals.framework import LLMJudge
    from routewise.models.client import chat
    from routewise.models.registry import display_name
    from routewise.studio.model import StudioBenchmark, load_saved
    from routewise.studio.report import html_report
    from routewise.studio.run import run_studio, summarize
    from routewise.studio.templates import STARTERS

    saved = load_saved()
    if args.studio_command == "list":
        print("| Benchmark | Name | Test cases | Checks |")
        print("|---|---|---|---|")
        for b in saved.values():
            checks = ", ".join(c.key for c in b.checks)
            print(f"| {b.slug} | {b.name} | {len(b.effective_cases())} | {checks} |")
        if not saved:
            print("No saved benchmarks yet. Build one in the dashboard's Benchmark studio.")
        print("\nTemplates (open them in the studio): " + ", ".join(t.name for t in STARTERS))
        return 0

    path = Path(args.benchmark)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        bench = StudioBenchmark.from_dict(yaml.safe_load(path.read_text()) or {})
    elif args.benchmark in saved:
        bench = saved[args.benchmark]
    else:
        print(
            f"unknown benchmark {args.benchmark!r} (see `routewise studio list`)", file=sys.stderr
        )
        return 2
    problems = bench.validate()
    if problems:
        print("The benchmark is not ready:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    registry = load_models(include_disabled=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    judge_model = args.judge or bench.judge_model or None
    unknown = [m for m in [*models, judge_model] if m and m not in registry]
    if unknown or not models:
        print(f"unknown models: {unknown or 'none given'}", file=sys.stderr)
        return 2
    if bench.uses_judge() and not judge_model:
        print("this benchmark uses an AI judge: pass --judge MODEL", file=sys.stderr)
        return 2

    store = ResultStore(args.db or default_db_path())
    chat_fn = functools.partial(chat, registry=registry)
    judge = LLMJudge(judge_model, chat_fn=chat_fn) if bench.uses_judge() else None
    summary = asyncio.run(
        run_studio(bench, models, store, chat_fn=chat_fn, judge=judge, max_cost=args.max_cost)
    )
    results = summarize(bench, models, store, registry)

    def name(model_id: str) -> str:
        return display_name(model_id, registry)

    print(f"\n{bench.name}: {results.total_cases} test cases, {len(models)} models")
    print("| Model | Score | Pass rate | Cost / case | Latency | Errors |")
    print("|---|---|---|---|---|---|")
    for m in results.models:
        print(
            f"| {name(m.model_id)} | {m.mean_score:.2f} | {m.pass_rate:.0%} | "
            f"${m.mean_cost_usd:.5f} | {m.mean_latency_s:.1f}s | {m.errors} |"
        )
    if results.best_quality:
        print(
            f"Best quality: {name(results.best_quality)}. Best value: {name(results.best_value)}."
        )
    if args.html:
        Path(args.html).write_text(html_report(bench, results, name))
        print(f"Report written to {args.html}")

    failed = summary.n_errors or summary.stopped_for_budget or summary.stopped_for_quota
    if args.min_score is not None:
        scored = {m.model_id: m.mean_score for m in results.models}
        below = [m for m in models if scored.get(m, 0.0) < args.min_score]
        if below:
            names = ", ".join(f"{name(m)} ({scored.get(m, 0.0):.2f})" for m in below)
            print(f"below the {args.min_score:.2f} bar: {names}", file=sys.stderr)
            return 1
    return 1 if failed else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("routewise.server:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routewise", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", help="results SQLite path (default data/results.db)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("models", help="list models").set_defaults(fn=cmd_models)
    sub.add_parser("eval", help="run evals (args go to the runner)")

    p = sub.add_parser("report", help="markdown results tables")
    p.add_argument("--out")
    p.add_argument("--modes", default="quality,balanced,cost")
    p.add_argument("--suite", help="limit to suites, e.g. core")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("route", help="route one query")
    p.add_argument("query")
    p.add_argument("--mode", default=None)
    p.add_argument("--execute", action="store_true", help="also call the routed model")
    p.set_defaults(fn=cmd_route)

    p = sub.add_parser("replay", help="offline router evaluation")
    p.add_argument("--mode", default="balanced")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("snapshot", help="save a quality baseline")
    p.add_argument("--out", default=str(DEFAULT_BASELINE))
    p.add_argument("--suite", help="limit the baseline to suites, e.g. core")
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("gate", help="fail if quality regressed vs baseline")
    p.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--min-n", type=int, default=2)
    p.add_argument("--suite", help="limit to suites; must match the baseline's scope")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="do not fail on baseline cells that have no current results",
    )
    p.set_defaults(fn=cmd_gate)

    sub.add_parser("rescore", help="rerun scorers on stored outputs").set_defaults(fn=cmd_rescore)
    p = sub.add_parser("audit", help="find suspect gold, saturated and flaky tasks")
    p.add_argument("--suite", help="limit to suites")
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("export", help="write results to JSONL (shareable, no secrets)")
    p.add_argument("--out", default="results/seed_results.jsonl")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("import", help="load results from a JSONL export")
    p.add_argument("path", nargs="?", default="results/seed_results.jsonl")
    p.set_defaults(fn=cmd_import)
    sub.add_parser("dashboard", help="launch Streamlit").set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("bench", help="blind, human-rated model comparisons")
    bench = p.add_subparsers(dest="bench_command", required=True)
    bench.add_parser("list", help="list benchmarks")
    b = bench.add_parser("run", help="generate blind responses for a new session")
    b.add_argument("--models", required=True, help="comma-separated model ids (2 or more)")
    b.add_argument("--baseline", required=True, help="the model you use today")
    b.add_argument("--suite", default="pm-core")
    b.add_argument("--bench", help="comma-separated benchmark ids (overrides --suite)")
    b.add_argument("--name")
    b = bench.add_parser("reveal", help="unblind a fully rated session")
    b.add_argument("session")
    bench.add_parser("history", help="human scores per model across sessions")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("studio", help="run benchmarks built in the Benchmark studio")
    studio = p.add_subparsers(dest="studio_command", required=True)
    studio.add_parser("list", help="list saved studio benchmarks")
    s = studio.add_parser("run", help="run a studio benchmark and print the results")
    s.add_argument("benchmark", help="a saved benchmark id, or a path to its YAML file")
    s.add_argument("--models", required=True, help="comma-separated model ids")
    s.add_argument("--judge", help="model for the AI judge check (default: the saved one)")
    s.add_argument("--max-cost", type=float, default=None, help="stop if spend reaches this ($)")
    s.add_argument("--min-score", type=float, default=None, help="exit 1 if any model scores below")
    s.add_argument("--html", help="also write a shareable HTML report here")
    p.set_defaults(fn=cmd_studio)

    p = sub.add_parser("serve", help="OpenAI-compatible routing endpoint")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":
        from routewise.evals.runner import main as runner_main

        return runner_main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
