"""Blind benchmarks page: rate answers to your own tasks without knowing which model wrote them,
then reveal which model you actually prefer."""

from __future__ import annotations

import asyncio
import functools

import altair as alt
import pandas as pd
import streamlit as st

from routewise.bench.core import (
    Benchmark,
    NotCompleteError,
    card_blurb,
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
from routewise.dashboard.narrative import reveal_headline, tied_top
from routewise.dashboard.ui import (
    ACCENT,
    GRAY_MARK,
    SEQ_RAMP,
    callout,
    chips,
    hero,
    inject_css,
    model_card,
    process,
    render,
    step,
    verdict_card,
)
from routewise.models.client import chat
from routewise.models.registry import (
    PROVIDER_LABELS,
    MissingCredentialsError,
    display_name,
    load_models,
    resolve_backend,
)

STEPS = {
    "setup": "1  Set up",
    "rate": "2  Rate blind",
    "reveal": "3  Reveal",
    "history": "History",
    "create": "Create a task",
}

inject_css()
store = BenchStore(default_db_path())
benches = load_benchmarks()
registry = load_models(include_disabled=True)


def name(model_id: str) -> str:
    return display_name(model_id, registry)


def provider(model_id: str) -> str:
    spec = registry.get(model_id)
    return PROVIDER_LABELS.get(spec.provider, spec.provider) if spec else ""


def _reachable(model_id: str) -> bool:
    try:
        resolve_backend(registry[model_id])
        return True
    except MissingCredentialsError:
        return False


def _go(step_key: str) -> None:
    st.session_state["bench-goto"] = step_key
    st.rerun()


available = [m for m, spec in registry.items() if spec.enabled and _reachable(m)]
sessions = store.sessions()
open_sessions = [s for s in sessions if not s.revealed_at]
revealed = [s for s in sessions if s.revealed_at]

if "bench-goto" in st.session_state:
    st.session_state["bench-step"] = st.session_state.pop("bench-goto")

render(
    hero(
        "Blind benchmarks",
        "Which model do you actually prefer?",
        "Public benchmarks measure models on someone else's tasks. Here you run your own tasks "
        "through several models, rate every answer without knowing which model wrote it, and "
        "only then see who wrote what. Tasks stay fixed, so you can rerun them whenever a new "
        "model ships and compare it with the model you use today.",
    )
)

current = st.radio(
    "Step",
    list(STEPS),
    format_func=STEPS.get,
    horizontal=True,
    key="bench-step",
    label_visibility="collapsed",
)
done = set()
if open_sessions or revealed:
    done.add(0)
if revealed:
    done.update({1, 2})
render(
    process(
        [
            ("Set up", "Pick tasks and the models to compare, including your current one."),
            ("Rate blind", "Score each answer 1 to 5 and say what you liked or disliked."),
            ("Reveal", "See which model wrote what, and which one you preferred."),
        ],
        active={"setup": 0, "rate": 1, "reveal": 2}.get(current, -1),
        done=done,
    )
)

# --- 1. set up ------------------------------------------------------------------------------

if current == "setup":
    if open_sessions:
        s = open_sessions[-1]
        rated, total = session_progress(store, s.session_id)
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(
            callout(
                f"You have an unfinished session, “{s.name or s.session_id}”: {rated} of "
                f"{total} answers rated.",
                "accent",
            ),
            unsafe_allow_html=True,
        )
        if c2.button("Continue rating", key="continue", use_container_width=True):
            st.session_state["bench-session"] = s.session_id
            _go("rate")

    render(
        step(
            1,
            "Choose the tasks",
            "Each task is a realistic scenario with fixed wording. "
            "Every model answers every task you pick.",
        )
    )
    suites = sorted({b.suite for b in benches.values()})
    picked = []
    for suite in suites:
        items = [b for b in benches.values() if b.suite == suite]
        st.markdown(
            f"**{'Product management' if suite == 'pm-core' else suite}** · {len(items)} tasks"
        )
        cols = st.columns(3, gap="medium")
        for i, b in enumerate(items):
            with cols[i % 3], st.container(border=True):
                if st.checkbox(b.title, value=suite == "pm-core", key=f"pick-{b.id}"):
                    picked.append(b.id)
                blurb = card_blurb(b)
                if blurb:
                    st.caption(blurb)
                if b.tags:
                    render(chips([(t, "") for t in b.tags]))

    render(
        step(
            2,
            "Choose the models",
            "Include the model you use today as the baseline. Names are hidden while you rate.",
        )
    )
    if len(available) < 2:
        render(
            callout(
                "You need at least two models you can call. Add an API key to .env "
                "(see .env.example) or install the claude CLI.",
                "accent",
            )
        )
    chosen_models = st.multiselect(
        "Models to compare",
        available,
        default=available[: min(3, len(available))],
        key="model-select",
        format_func=name,
    )
    baseline = st.selectbox(
        "Which one do you use today?",
        chosen_models or ["(pick models first)"],
        key="baseline-select",
        format_func=lambda m: name(m) if m in registry else m,
    )
    session_name = st.text_input(
        "Name this session (optional)",
        key="session-name",
        placeholder="e.g. GPT-6.1 Sol vs my current model, October",
    )

    n_calls = len(picked) * len(chosen_models)
    render(step(3, "Generate the answers"))
    render(
        callout(
            f"You will rate {n_calls} answers ({len(picked)} tasks × {len(chosen_models)} "
            f"models), about {max(1, round(n_calls * 1.2))} minutes. Answers are shuffled "
            "separately for each task, model and company names are masked, and cost and "
            "speed are hidden until the reveal.",
            "accent",
        )
    )
    st.markdown("")
    if st.button(
        "Generate blind answers",
        key="generate",
        type="primary",
        disabled=n_calls == 0 or len(chosen_models) < 2,
    ):
        try:
            sid = create_session(
                store,
                [benches[b] for b in picked],
                chosen_models,
                baseline=baseline,
                name=session_name or None,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            chat_fn = functools.partial(chat, registry=registry)
            with st.status(f"Generating {n_calls} answers...", expanded=False) as status:
                ok = asyncio.run(
                    generate_outputs(
                        store,
                        sid,
                        benches,
                        chat_fn=chat_fn,
                        progress=lambda line: status.write(line),
                    )
                )
                status.update(label=f"{ok} of {n_calls} answers ready", state="complete")
            st.session_state["bench-session"] = sid
            st.session_state["task-idx"] = 0
            _go("rate")

# --- 2. rate --------------------------------------------------------------------------------

elif current == "rate":
    if not open_sessions:
        render(callout("Nothing to rate yet. Start a session in step 1.", "accent"))
        if st.button("Go to set up", key="to-setup"):
            _go("setup")
        st.stop()

    ids = [s.session_id for s in reversed(open_sessions)]
    chosen_sid = st.session_state.get("bench-session")
    sid = st.selectbox(
        "Session",
        ids,
        index=ids.index(chosen_sid) if chosen_sid in ids else 0,
        format_func=lambda i: (
            f"{store.session(i).name or i}  "
            f"({session_progress(store, i)[0]} of {session_progress(store, i)[1]} rated)"
        ),
        key="rate-session",
    )
    session = store.session(sid)
    outputs = store.outputs(sid)
    rated, total = session_progress(store, sid)
    st.progress(rated / total if total else 0.0, text=f"{rated} of {total} answers rated")

    missing = [o for o in outputs if o.output is None]
    if missing:
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.warning(f"{len(missing)} answers failed to generate.")
        if c2.button("Retry", key="retry", use_container_width=True):
            chat_fn = functools.partial(chat, registry=registry)
            asyncio.run(generate_outputs(store, sid, benches, chat_fn=chat_fn))
            st.rerun()

    task_ids = session.benchmark_ids
    idx = min(st.session_state.get("task-idx", 0), len(task_ids) - 1)

    def task_done(bid: str) -> bool:
        return all(o.score is not None for o in outputs if o.benchmark_id == bid)

    def _label(i: int) -> str:
        b = benches.get(task_ids[i])
        mark = "  ✓" if task_done(task_ids[i]) else ""
        return f"Task {i + 1} of {len(task_ids)}: {b.title if b else task_ids[i]}{mark}"

    nav_l, nav_m, nav_r = st.columns([1, 4, 1], vertical_alignment="bottom")
    if nav_l.button("← Previous", key="prev-task", disabled=idx == 0, use_container_width=True):
        st.session_state["task-idx"] = idx - 1
        st.rerun()
    picked_idx = nav_m.selectbox(
        "Task",
        range(len(task_ids)),
        index=idx,
        format_func=_label,
        key=f"task-pick-{sid}-{idx}",
        label_visibility="collapsed",
    )
    if picked_idx != idx:
        st.session_state["task-idx"] = picked_idx
        st.rerun()
    if nav_r.button(
        "Next →", key="next-task", disabled=idx >= len(task_ids) - 1, use_container_width=True
    ):
        st.session_state["task-idx"] = idx + 1
        st.rerun()

    bid = task_ids[idx]
    bench = benches.get(bid)
    render(
        step(
            idx + 1,
            bench.title if bench else bid,
            "Read the task, then score each answer on its own merits.",
        )
    )
    if bench:
        with st.expander("The task, exactly as the models saw it", expanded=False):
            st.text(bench.prompt)
        if bench.guidance:
            render(callout(f"What a great answer does: {bench.guidance}", "accent"))
        st.markdown("")

    task_outputs = sorted((o for o in outputs if o.benchmark_id == bid), key=lambda o: o.label)
    per_row = min(3, max(1, len(task_outputs)))
    for start in range(0, len(task_outputs), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, o in zip(cols, task_outputs[start : start + per_row], strict=False):
            with col, st.container(border=True):
                status_txt = "  ✓ rated" if o.score is not None else ""
                st.markdown(
                    f'<div class="rw-resp-label">Response {o.label}{status_txt}</div>',
                    unsafe_allow_html=True,
                )
                if o.output is None:
                    st.caption("Not generated yet.")
                    continue
                with st.container(height=440, border=False):
                    st.markdown(demote_headings(escape_markdown(redact_identity(o.output))))
                with st.form(f"form-{o.id}", border=False):
                    score = st.radio(
                        "Your score (1 poor, 5 excellent)",
                        [1, 2, 3, 4, 5],
                        index=(o.score - 1) if o.score else None,
                        horizontal=True,
                        key=f"score-{o.id}",
                    )
                    liked = st.text_input(
                        "What you liked",
                        value=o.liked or "",
                        key=f"liked-{o.id}",
                        placeholder="e.g. clear plan, respected the budget",
                    )
                    disliked = st.text_input(
                        "What you disliked",
                        value=o.disliked or "",
                        key=f"disliked-{o.id}",
                        placeholder="e.g. invented a number",
                    )
                    if st.form_submit_button(
                        "Save rating", key=f"save-{o.id}", use_container_width=True
                    ):
                        if score is None:
                            st.error("Pick a score from 1 to 5.")
                        else:
                            try:
                                rate(store, o.id, score, liked, disliked)
                            except ValueError as exc:
                                st.error(str(exc))
                            else:
                                st.rerun()

    if task_done(bid) and idx < len(task_ids) - 1:
        st.success("All answers for this task are rated.")
        if st.button("Next task →", key="next-task-cta", type="primary"):
            st.session_state["task-idx"] = idx + 1
            st.rerun()

    st.divider()
    rated, total = session_progress(store, sid)
    left = total - rated
    render(
        callout(
            "Every answer is rated. Reveal which model wrote what."
            if not left
            else f"{left} answer{'s' if left != 1 else ''} left to rate before you can reveal "
            "the models.",
            "accent" if not left else "",
        )
    )
    st.markdown("")
    if st.button("Reveal the models", key="reveal", type="primary", disabled=bool(left)):
        try:
            reveal(store, sid)
        except NotCompleteError as exc:
            st.error(str(exc))
        else:
            st.session_state["revealed"] = sid
            _go("reveal")

# --- 3. reveal ------------------------------------------------------------------------------

elif current == "reveal":
    if not revealed:
        render(
            callout(
                "Results appear here once you rate every answer in a session and reveal it.",
                "accent",
            )
        )
        st.stop()
    ids = [s.session_id for s in reversed(revealed)]
    last = st.session_state.get("revealed")
    sid = st.selectbox(
        "Session",
        ids,
        index=ids.index(last) if last in ids else 0,
        format_func=lambda i: store.session(i).name or i,
        key="results-session",
    )
    session = store.session(sid)
    result = reveal(store, sid)
    n_tasks = len(session.benchmark_ids)
    head, line = reveal_headline(result, name, n_tasks=n_tasks)
    render(
        verdict_card(
            head,
            line,
            [
                (f"{n_tasks} tasks", ""),
                (f"{len(result.models)} models", ""),
                (f"Your current model: {name(result.baseline)}", "accent"),
            ],
        )
    )

    render(
        step(
            "1",
            "How each model did",
            "Average of your scores, tasks won (ties split), and "
            "the gap to the model you use today.",
        )
    )
    leaders = tied_top(result)
    cols = st.columns(min(3, len(result.models)), gap="medium")
    for i, m in enumerate(result.models):
        if len(leaders) > 1 and m.model_id in leaders:
            badge, bkind, kind = "Tied for first", "accent", "chosen"
        elif m.model_id == result.preferred:
            badge, bkind, kind = "Your pick", "accent", "chosen"
        elif m.is_baseline:
            badge, bkind, kind = "Your current model", "", ""
        else:
            badge, bkind, kind = f"#{i + 1}", "", ""
        delta = "baseline" if m.is_baseline else f"{m.delta_vs_baseline:+.1f}"
        cols[i % len(cols)].markdown(
            model_card(
                name=name(m.model_id),
                provider=provider(m.model_id),
                badge=badge,
                badge_kind=bkind,
                kind=kind,
                stats=[
                    ("Average score", f"{m.mean_score:.1f} / 5", m.mean_score / 5),
                    ("Tasks won", f"{m.wins:g} of {n_tasks}", m.wins / n_tasks if n_tasks else 0),
                    ("vs. your current model", delta, None),
                ],
                footnote=f"{m.n} answers rated.",
            ),
            unsafe_allow_html=True,
        )

    render(
        step("2", "Task by task", "Your score for every answer. ★ marks the winner of each task.")
    )
    rows = pd.DataFrame(
        [
            {
                "Task": benches[r.benchmark_id].title
                if r.benchmark_id in benches
                else r.benchmark_id,
                "Model": name(r.model_id),
                "Score": r.score,
                "Answer": r.label,
                "Liked": r.liked,
                "Disliked": r.disliked,
            }
            for r in result.rows
        ]
    )
    rows["Best"] = rows.groupby("Task")["Score"].transform("max") == rows["Score"]
    rows["Label"] = rows["Score"].astype(str) + rows["Best"].map({True: " ★", False: ""})
    model_order = [name(m.model_id) for m in result.models]
    grid = alt.Chart(rows).encode(
        x=alt.X(
            "Model:N",
            sort=model_order,
            title=None,
            axis=alt.Axis(
                labelAngle=0,
                orient="top",
                labelFontSize=13,
                ticks=False,
                domain=False,
                labelLimit=200,
            ),
        ),
        y=alt.Y(
            "Task:N",
            title=None,
            axis=alt.Axis(labelFontSize=12, ticks=False, domain=False, labelLimit=320),
        ),
    )
    cells = grid.mark_rect(cornerRadius=4, stroke="#fcfcfb", strokeWidth=2).encode(
        color=alt.Color(
            "Score:Q",
            scale=alt.Scale(domain=[1, 5], range=SEQ_RAMP[:6]),
            legend=alt.Legend(title="Score", orient="right", gradientLength=110, values=[1, 3, 5]),
        ),
        tooltip=["Task", "Model", "Answer", "Score", "Liked", "Disliked"],
    )
    text = grid.mark_text(fontSize=13, fontWeight=600).encode(
        text="Label:N",
        color=alt.condition("datum.Score >= 3.5", alt.value("white"), alt.value("#0b0b0b")),
    )
    st.altair_chart((cells + text).properties(height=alt.Step(40)), width="stretch")

    render(step("3", "What you said", "Your comments, grouped by model."))
    cols = st.columns(min(3, len(result.models)), gap="medium")
    for i, m in enumerate(result.models):
        with cols[i % len(cols)], st.container(border=True):
            st.markdown(f"**{name(m.model_id)}**")
            for t in m.liked:
                st.markdown(f":green[**+**] {escape_markdown(t)}")
            for t in m.disliked:
                st.markdown(f":red[**−**] {escape_markdown(t)}")
            if not m.liked and not m.disliked:
                st.caption("No comments.")
    st.download_button(
        "Download these results (CSV)",
        rows.drop(columns=["Best", "Label"]).to_csv(index=False).encode(),
        file_name=f"blind-benchmark-{sid}.csv",
        mime="text/csv",
        key="download",
    )

# --- history ------------------------------------------------------------------------------

elif current == "history":
    render(
        step(
            "★",
            "Your verdicts over time",
            "Every revealed session, combined. Only ratings "
            "on the current wording of each task count, so new and old models are compared "
            "on exactly the same work.",
        )
    )
    hist = history(store, benches)
    if not hist:
        render(
            callout(
                "No revealed sessions yet. Rate and reveal a session to start your history.",
                "accent",
            )
        )
    else:
        frame = pd.DataFrame(
            [
                {
                    "Model": name(h.model_id),
                    "Average score": round(h.mean_score, 2),
                    "Ratings": h.ratings,
                    "Sessions": h.sessions,
                    "First rated": h.first_at[:10],
                    "Last rated": h.last_at[:10],
                }
                for h in hist
            ]
        )
        top = frame["Model"].iloc[0]
        frame["Highlight"] = frame["Model"].map(lambda m: "Top" if m == top else "Other")
        bars = alt.Chart(frame).encode(
            y=alt.Y(
                "Model:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelFontSize=13, ticks=False, domain=False, labelLimit=220),
            ),
            x=alt.X(
                "Average score:Q",
                scale=alt.Scale(domain=[0, 5]),
                title="Average score (1-5)",
                axis=alt.Axis(grid=True, gridColor="#eeede9", domain=False),
            ),
        )
        chart = bars.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=22).encode(
            color=alt.Color(
                "Highlight:N",
                scale=alt.Scale(domain=["Top", "Other"], range=[ACCENT, GRAY_MARK]),
                legend=None,
            ),
            tooltip=["Model", "Average score", "Ratings", "Sessions", "First rated", "Last rated"],
        ) + bars.mark_text(align="left", dx=6, fontSize=12, color="#0b0b0b").encode(
            text=alt.Text("Average score:Q", format=".2f")
        )
        st.altair_chart(chart.properties(height=alt.Step(38)), width="stretch")
        st.dataframe(frame.drop(columns=["Highlight"]), hide_index=True, width="stretch")

# --- create -------------------------------------------------------------------------------

elif current == "create":
    render(
        step(
            "+",
            "Create a task from your own work",
            "Write it once and keep it fixed. Every future model gets exactly the same task, "
            "so your comparisons stay fair over time.",
        )
    )
    left, right = st.columns([3, 2], gap="large")
    with left:
        title = st.text_input(
            "Title",
            key="new-title",
            placeholder="e.g. Turn my notes into a quarterly planning memo",
        )
        prompt = st.text_area(
            "The task, exactly as the models will see it",
            height=220,
            key="new-prompt",
            placeholder="Paste a real request plus the context a colleague "
            "would need: data, constraints, the format you want.",
        )
        guidance = st.text_area(
            "What a great answer does (only you see this, while rating)",
            key="new-guidance",
            height=100,
            placeholder="e.g. Makes explicit cuts, ties every priority to a "
            "metric, stays under one page.",
        )
        with st.expander("Advanced"):
            suite = st.text_input("Group (suite)", value="my-benchmarks", key="new-suite")
            system_prompt = st.text_area("System prompt (optional)", key="new-system")
        if st.button("Save task", key="new-save", type="primary"):
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
                st.success(f"Saved. “{title}” is now available in step 1.")
                st.caption(f"Stored at {path} (gitignored, so it stays private).")
    with right:
        render(
            callout(
                "Good tasks are specific and self-contained: include the numbers, constraints and "
                "audience a colleague would need. Avoid tasks with one right word as the answer; "
                "blind rating shines on judgement calls like plans, writing and trade-offs.",
                "accent",
            )
        )
        mine = [b for b in benches.values() if b.suite != "pm-core"]
        st.markdown("")
        st.markdown(f"**Your tasks** ({len(mine)})")
        if not mine:
            st.caption(f"None yet. They are saved in {custom_dir()}.")
        for b in mine:
            st.markdown(f"- {b.title}  \n  :gray[{b.suite} · {b.id}]")
