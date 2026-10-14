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
from routewise.router.leaderboard import Leaderboard
from routewise.router.router import Router, load_router_config

DEFAULT_BASELINE = PROJECT_ROOT / "results" / "baseline.json"


class Context:
    """Lazily loaded store, tasks, registry, leaderboard, and router."""

    def __init__(self, db: str | None):
        self.store = ResultStore(db or default_db_path())
        self.tasks = all_tasks()
        self.registry = load_models(include_disabled=True)
        self.config = load_router_config()
        self.board = Leaderboard.from_store(self.store, current=self.tasks)
        self.router = Router(self.board, self.config, self.registry)


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
    ctx = Context(args.db)
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

    ctx = Context(args.db)
    save_snapshot(snapshot(ctx.board), args.out)
    print(f"wrote baseline {args.out}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from routewise.router.gate import gate, load_snapshot

    ctx = Context(args.db)
    result = gate(
        ctx.board, load_snapshot(args.baseline), threshold=args.threshold, min_n=args.min_n
    )
    print(result.summary())
    return 0 if result.passed else 1


def cmd_rescore(args: argparse.Namespace) -> int:
    from routewise.evals.rescore import rescore

    ctx = Context(args.db)
    changed = asyncio.run(rescore(ctx.store, ctx.tasks))
    print(f"rescored {changed} rows")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    app = Path(__file__).parent / "dashboard" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


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
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("gate", help="fail if quality regressed vs baseline")
    p.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--min-n", type=int, default=2)
    p.set_defaults(fn=cmd_gate)

    sub.add_parser("rescore", help="rerun scorers on stored outputs").set_defaults(fn=cmd_rescore)
    sub.add_parser("dashboard", help="launch Streamlit").set_defaults(fn=cmd_dashboard)

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
