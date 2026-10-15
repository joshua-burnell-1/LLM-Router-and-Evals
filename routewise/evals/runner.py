"""Run every eval task against every model and store one row per (task, model, sample).

    uv run python -m routewise.evals.runner --limit 4            # cheap smoke run
    uv run python -m routewise.evals.runner                      # everything
    uv run python -m routewise.evals.runner --models claude-haiku-5-5 --suite core --repeats 3

Reruns skip pairs already recorded (unless --force). Errored rows are retried on rerun.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import sys
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass

from dotenv import load_dotenv

from routewise.config import TASKS_DIR, default_db_path
from routewise.evals.framework import EvalTask, LLMJudge, score_detail_json, score_output
from routewise.evals.loader import all_tasks
from routewise.evals.store import ResultRow, ResultStore, now_iso
from routewise.models.client import ChatResult, QuotaExhaustedError, chat
from routewise.models.registry import MissingCredentialsError, load_models, resolve_backend

DEFAULT_JUDGE_MODEL = "claude-haiku-5-5"


@dataclass
class RunSummary:
    run_id: str
    n_done: int = 0
    n_skipped: int = 0
    n_errors: int = 0
    total_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    stopped_for_budget: bool = False
    stopped_for_quota: bool = False
    n_needs_judge: int = 0


async def run_eval(
    tasks: Sequence[EvalTask],
    model_ids: Sequence[str],
    store: ResultStore,
    *,
    judge: LLMJudge | None,
    repeats: int = 1,
    concurrency: int = 4,
    force: bool = False,
    max_cost: float | None = None,
    chat_fn: Callable[..., Awaitable[ChatResult]] = chat,
    progress: Callable[[str], None] = print,
    run_args: dict | None = None,
) -> RunSummary:
    summary = RunSummary(run_id=store.start_run(run_args or {}))
    jobs = []
    for task in tasks:
        if judge is None and any(s.type == "llm_judge" for s in task.scoring):
            # Scoring without the judge would store a partial score as if it were complete.
            summary.n_needs_judge += 1
            continue
        for model_id in model_ids:
            for sample in range(repeats):
                if not force and store.has(task.id, task.version, model_id, sample):
                    summary.n_skipped += 1
                else:
                    jobs.append((task, model_id, sample))

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    counter = {"n": 0}
    total = len(jobs)

    async def run_one(task: EvalTask, model_id: str, sample: int) -> None:
        async with semaphore:
            async with lock:
                if summary.stopped_for_quota:
                    return
                if max_cost is not None and summary.total_cost_usd >= max_cost:
                    summary.stopped_for_budget = True
                    return
            row = ResultRow(
                task_id=task.id,
                task_version=task.version,
                model_id=model_id,
                created_at=now_iso(),
                sample_idx=sample,
                run_id=summary.run_id,
                suite=task.suite,
                category=task.category,
                judge_model=judge.model_id if judge else None,
            )
            try:
                result = await chat_fn(model_id, task.messages(), max_tokens=task.max_tokens)
                row.output = result.text
                row.latency_s = result.latency_s
                row.input_tokens = result.input_tokens
                row.output_tokens = result.output_tokens
                row.tokens_estimated = result.tokens_estimated
                row.cost_usd = result.cost_usd
                row.backend = result.backend
                async with lock:
                    summary.total_cost_usd += result.cost_usd
                scored = await score_output(task, result.text, judge)
                row.quality = scored.value
                row.score_detail = score_detail_json(scored.components)
                row.judge_cost_usd = scored.judge_cost_usd
                async with lock:
                    summary.total_cost_usd += scored.judge_cost_usd
                    summary.judge_cost_usd += scored.judge_cost_usd
            except QuotaExhaustedError as exc:  # plan limit hit: stop starting new calls
                summary.stopped_for_quota = True
                row.error = f"{type(exc).__name__}: {exc}"[:2000]
                row.quality = None
            except Exception as exc:  # recorded, retried on the next run
                row.error = f"{type(exc).__name__}: {exc}"[:2000]
                row.quality = None
            store.record(row)
            async with lock:
                counter["n"] += 1
                k = counter["n"]
                if row.error:
                    summary.n_errors += 1
                    progress(f"[{k}/{total}] {task.id} x {model_id} ERROR {row.error[:120]}")
                else:
                    summary.n_done += 1
                    progress(
                        f"[{k}/{total}] {task.id} x {model_id} q={row.quality:.2f} "
                        f"{row.latency_s:.1f}s ${row.cost_usd:.5f}"
                    )

    await asyncio.gather(*(run_one(*job) for job in jobs))
    store.finish_run(summary.run_id, summary.total_cost_usd, summary.n_done + summary.n_errors)
    return summary


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run routewise evals and store results in SQLite.")
    p.add_argument("--limit", type=int, help="only the first N tasks (cheap iteration)")
    p.add_argument("--models", help="comma-separated model ids (default: all enabled)")
    p.add_argument("--suite", help="comma-separated suites, e.g. core,scale-triage")
    p.add_argument("--category", help="comma-separated categories")
    p.add_argument("--tag", help="comma-separated tags")
    p.add_argument("--task", help="comma-separated task ids")
    p.add_argument("--no-packs", action="store_true", help="built-in 16 tasks only")
    p.add_argument("--repeats", type=int, default=1, help="samples per (task, model)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--force", action="store_true", help="rerun pairs already recorded")
    p.add_argument("--max-cost", type=float, help="stop starting new calls past this USD")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--no-judge", action="store_true", help="programmatic scorers only")
    p.add_argument("--db", help="SQLite path (default data/results.db or $ROUTEWISE_DB)")
    p.add_argument("--dry-run", action="store_true", help="list the jobs, call nothing")
    return p


def select_tasks(args: argparse.Namespace) -> list[EvalTask]:
    tasks = all_tasks(
        packs_dir=None if args.no_packs else TASKS_DIR,
        suites=_csv(args.suite),
        categories=_csv(args.category),
        tags=_csv(args.tag),
        ids=_csv(args.task),
    )
    return tasks[: args.limit] if args.limit else tasks


def main(argv: Iterable[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    registry = load_models(include_disabled=True)
    explicit = _csv(args.models)
    unknown = [m for m in explicit or [] if m not in registry]
    if unknown:
        print(f"unknown models: {unknown}; known: {sorted(registry)}", file=sys.stderr)
        return 2
    model_ids = []
    for model_id in explicit or [m for m, spec in registry.items() if spec.enabled]:
        try:
            resolve_backend(registry[model_id])
        except MissingCredentialsError as exc:
            if explicit:  # asked for by name: say why it cannot run
                print(str(exc), file=sys.stderr)
                return 2
            print(f"skipping {model_id}: {exc}")
            continue
        model_ids.append(model_id)
    if not model_ids:
        print("no reachable models: add an API key to .env (see .env.example)", file=sys.stderr)
        return 2
    tasks = select_tasks(args)
    if not tasks:
        print("no tasks match the filters", file=sys.stderr)
        return 2
    print(f"{len(tasks)} tasks x {len(model_ids)} models x {args.repeats} samples")
    if args.dry_run:
        for task in tasks:
            print(f"  {task.id} [{task.suite}/{task.category}]")
        return 0

    chat_fn = functools.partial(chat, registry=registry)
    judge = None
    if not args.no_judge:
        if args.judge_model not in registry:
            print(f"judge model {args.judge_model!r} not in registry", file=sys.stderr)
            return 2
        judge = LLMJudge(args.judge_model, chat_fn=chat_fn)
    store = ResultStore(args.db or default_db_path())
    summary = asyncio.run(
        run_eval(
            tasks,
            model_ids,
            store,
            judge=judge,
            repeats=args.repeats,
            concurrency=args.concurrency,
            force=args.force,
            max_cost=args.max_cost,
            chat_fn=chat_fn,
            run_args=vars(args),
        )
    )
    print(
        f"run {summary.run_id}: {summary.n_done} done, {summary.n_skipped} skipped, "
        f"{summary.n_errors} errors, ${summary.total_cost_usd:.4f} "
        f"(judge ${summary.judge_cost_usd:.4f})"
        + (" [stopped at budget cap]" if summary.stopped_for_budget else "")
        + (
            " [stopped: plan or credit limit reached; rerun later]"
            if summary.stopped_for_quota
            else ""
        )
        + (
            f" [{summary.n_needs_judge} judged tasks skipped by --no-judge]"
            if summary.n_needs_judge
            else ""
        )
    )
    if summary.n_errors:
        return 1
    return 3 if summary.stopped_for_budget or summary.stopped_for_quota else 0


if __name__ == "__main__":
    raise SystemExit(main())
