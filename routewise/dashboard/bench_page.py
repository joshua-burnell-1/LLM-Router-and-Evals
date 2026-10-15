"""Blind benchmarks page: rate responses without knowing which model wrote them."""

from __future__ import annotations

import asyncio
import functools

import pandas as pd
import streamlit as st

from routewise.bench.core import (
    Benchmark,
    NotCompleteError,
    create_session,
    custom_dir,
    generate_outputs,
    history,
    load_benchmarks,
    rate,
    redact_identity,
    reveal,
    save_custom_benchmark,
    session_progress,
)
from routewise.bench.store import BenchStore
from routewise.config import default_db_path
from routewise.dashboard import demote_headings, escape_markdown
from routewise.models.client import chat
from routewise.models.registry import MissingCredentialsError, load_models, resolve_backend

store = BenchStore(default_db_path())
benches = load_benchmarks()
registry = load_models(include_disabled=True)


def _reachable(model_id: str) -> bool:
    try:
        resolve_backend(registry[model_id])
        return True
    except MissingCredentialsError:
        return False


available = [m for m, spec in registry.items() if spec.enabled and _reachable(m)]

st.title("Blind benchmarks")
st.caption(
    "Run the same tasks across models, rate each response without knowing which model wrote "
    "it, then reveal which model you actually preferred. Prompts are fixed and versioned, so "
    "you can rerun them whenever a new model ships and compare against your current model."
)

start_tab, rate_tab, results_tab, create_tab = st.tabs(
    ["1. Start a session", "2. Rate responses", "3. Results", "Create a benchmark"]
)

# --- start ------------------------------------------------------------------------------------

with start_tab:
    suites = sorted({b.suite for b in benches.values()})
    default_suite = "pm-core" if "pm-core" in suites else (suites[0] if suites else None)
    chosen_ids = st.multiselect(
        "Benchmarks",
        list(benches),
        default=[b.id for b in benches.values() if b.suite == default_suite],
        format_func=lambda b: f"{benches[b].title} ({benches[b].suite})",
        key="bench-select",
    )
    chosen_models = st.multiselect(
        "Models to compare (2 or more)",
        available,
        default=available[: min(3, len(available))],
        key="model-select",
        help="Only models you can call (API key set, or the claude CLI for Claude) are listed.",
    )
    baseline = st.selectbox(
        "Your current model (the baseline)",
        chosen_models or ["(pick models first)"],
        key="baseline-select",
    )
    name = st.text_input("Session name (optional)", key="session-name")
    n_calls = len(chosen_ids) * len(chosen_models)
    st.caption(f"{n_calls} responses will be generated. Model names stay hidden until you reveal.")
    if st.button(
        "Generate blind responses",
        key="generate",
        type="primary",
        disabled=n_calls == 0 or len(chosen_models) < 2,
    ):
        try:
            sid = create_session(
                store,
                [benches[b] for b in chosen_ids],
                chosen_models,
                baseline=baseline,
                name=name or None,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            chat_fn = functools.partial(chat, registry=registry)
            with st.status(f"Generating {n_calls} responses...", expanded=False) as status:
                done = asyncio.run(
                    generate_outputs(
                        store,
                        sid,
                        benches,
                        chat_fn=chat_fn,
                        progress=lambda line: status.write(line),
                    )
                )
                status.update(label=f"{done} of {n_calls} responses ready", state="complete")
            st.session_state["bench-session"] = sid
            st.success("Ready. Open the '2. Rate responses' tab.")

# --- rate -------------------------------------------------------------------------------------

open_sessions = [s for s in store.sessions() if not s.revealed_at]

with rate_tab:
    if not open_sessions:
        st.info("No sessions waiting for ratings. Start one in the first tab.")
    else:
        ids = [s.session_id for s in reversed(open_sessions)]
        current = st.session_state.get("bench-session")
        sid = st.selectbox(
            "Session",
            ids,
            index=ids.index(current) if current in ids else 0,
            format_func=lambda i: (
                f"{store.session(i).name or i} "
                f"({session_progress(store, i)[0]}/{session_progress(store, i)[1]} rated)"
            ),
            key="rate-session",
        )
        session = store.session(sid)
        rated, total = session_progress(store, sid)
        st.progress(rated / total if total else 0.0, text=f"{rated} of {total} responses rated")

        outputs = store.outputs(sid)
        missing = [o for o in outputs if o.output is None]
        if missing:
            st.warning(f"{len(missing)} responses failed to generate.")
            if st.button("Retry missing responses", key="retry"):
                chat_fn = functools.partial(chat, registry=registry)
                asyncio.run(generate_outputs(store, sid, benches, chat_fn=chat_fn))
                st.rerun()

        for bench_id in session.benchmark_ids:
            bench = benches.get(bench_id)
            st.subheader(bench.title if bench else bench_id)
            if bench:
                with st.expander("Task prompt"):
                    st.text(bench.prompt)
                if bench.guidance:
                    st.caption(f"What good looks like: {bench.guidance}")
            for o in sorted(
                (o for o in outputs if o.benchmark_id == bench_id), key=lambda o: o.label
            ):
                with st.container(border=True):
                    done_mark = " (rated)" if o.score is not None else ""
                    st.markdown(f"**Response {o.label}**{done_mark}")
                    if o.output is None:
                        st.caption("Not generated yet.")
                        continue
                    with st.container(height=420, border=False):
                        st.markdown(demote_headings(escape_markdown(redact_identity(o.output))))
                    with st.form(f"form-{o.id}", border=False):
                        score = st.radio(
                            "Score (1 = poor, 5 = excellent)",
                            [1, 2, 3, 4, 5],
                            index=(o.score - 1) if o.score else None,
                            horizontal=True,
                            key=f"score-{o.id}",
                        )
                        liked = st.text_input(
                            "What you liked", value=o.liked or "", key=f"liked-{o.id}"
                        )
                        disliked = st.text_input(
                            "What you disliked", value=o.disliked or "", key=f"disliked-{o.id}"
                        )
                        if st.form_submit_button("Save rating", key=f"save-{o.id}"):
                            if score is None:
                                st.error("Pick a score from 1 to 5.")
                            else:
                                try:
                                    rate(store, o.id, score, liked, disliked)
                                except ValueError as exc:
                                    st.error(str(exc))
                                else:
                                    st.rerun()

        rated, total = session_progress(store, sid)
        st.divider()
        if st.button(
            "Reveal models",
            key="reveal",
            type="primary",
            disabled=rated < total,
            help="Available once every response has a rating.",
        ):
            try:
                result = reveal(store, sid)
            except NotCompleteError as exc:
                st.error(str(exc))
            else:
                st.session_state["revealed"] = sid
                st.success(
                    f"You preferred {result.preferred} "
                    f"(mean {result.models[0].mean_score:.2f}). See the Results tab."
                )

# --- results ----------------------------------------------------------------------------------

with results_tab:
    revealed = [s for s in store.sessions() if s.revealed_at]
    if not revealed:
        st.info("Results appear here once you reveal a fully rated session.")
    else:
        ids = [s.session_id for s in reversed(revealed)]
        last = st.session_state.get("revealed")
        sid = st.selectbox(
            "Revealed session",
            ids,
            index=ids.index(last) if last in ids else 0,
            format_func=lambda i: store.session(i).name or i,
            key="results-session",
        )
        result = reveal(store, sid)
        st.success(f"You preferred **{result.preferred}**. Your baseline was {result.baseline}.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "model": m.model_id + (" (baseline)" if m.is_baseline else ""),
                        "mean score": round(m.mean_score, 2),
                        "prompt wins": m.wins,
                        "vs baseline": round(m.delta_vs_baseline, 2),
                        "ratings": m.n,
                    }
                    for m in result.models
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**By benchmark** (which label was which model)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "benchmark": benches[r.benchmark_id].title
                        if r.benchmark_id in benches
                        else r.benchmark_id,
                        "response": r.label,
                        "model": r.model_id,
                        "score": r.score,
                        "liked": r.liked,
                        "disliked": r.disliked,
                    }
                    for r in result.rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        for m in result.models:
            with st.expander(f"Your comments on {m.model_id}"):
                for text in m.liked:
                    st.markdown(f"- Liked: {escape_markdown(text)}")
                for text in m.disliked:
                    st.markdown(f"- Disliked: {escape_markdown(text)}")

    st.subheader("History across sessions")
    rows = history(store, benches)
    if not rows:
        st.caption("No revealed sessions yet.")
    else:
        frame = pd.DataFrame(
            [
                {
                    "model": r.model_id,
                    "mean human score": round(r.mean_score, 2),
                    "ratings": r.ratings,
                    "sessions": r.sessions,
                    "first": r.first_at[:10],
                    "last": r.last_at[:10],
                }
                for r in rows
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch")
        st.bar_chart(frame.set_index("model")["mean human score"])
        st.caption("Only ratings on the current version of each benchmark prompt are counted.")

# --- create -----------------------------------------------------------------------------------

with create_tab:
    st.markdown(
        "Write a task from your own work. Keep the prompt fixed so future models can be "
        f"compared on exactly the same task. Saved to `{custom_dir()}` (gitignored by default)."
    )
    title = st.text_input("Title", key="new-title")
    suite = st.text_input("Suite", value="my-benchmarks", key="new-suite")
    prompt = st.text_area("Prompt (exactly what the models will see)", height=200, key="new-prompt")
    guidance = st.text_area(
        "What a great answer does (shown only to you while rating)", key="new-guidance"
    )
    system_prompt = st.text_area("System prompt (optional)", key="new-system")
    if st.button("Save benchmark", key="new-save", type="primary"):
        try:
            path = save_custom_benchmark(
                Benchmark(
                    id="",
                    title=title,
                    prompt=prompt,
                    suite=suite or "my-benchmarks",
                    guidance=guidance,
                    system_prompt=system_prompt or None,
                )
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(f"Saved {path.name}. It is now available in '1. Start a session'.")
    mine = [b for b in benches.values() if b.suite != "pm-core"]
    if mine:
        st.markdown("**Your benchmarks**")
        for b in mine:
            st.markdown(f"- {b.title} (`{b.id}`, suite {b.suite})")
