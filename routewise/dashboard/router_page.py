"""Model router page: enter a prompt, get a recommendation, see how every model compares."""

from __future__ import annotations

import asyncio
import functools
import os

import altair as alt
import pandas as pd
import streamlit as st

from routewise.config import default_db_path
from routewise.dashboard import demote_headings, escape_markdown
from routewise.dashboard.narrative import (
    CATEGORY_BLURBS,
    MODES,
    category_label,
    detection_note,
    verdict,
)
from routewise.dashboard.ui import (
    ACCENT,
    GRAY_MARK,
    SEQ_RAMP,
    callout,
    hero,
    inject_css,
    kpis,
    model_card,
    money,
    render,
    step,
    verdict_card,
)
from routewise.evals.loader import all_tasks
from routewise.evals.store import ResultStore
from routewise.models.client import chat
from routewise.models.registry import PROVIDER_LABELS, display_name, load_models
from routewise.router.classifier import ExampleIndex
from routewise.router.leaderboard import OVERALL, Leaderboard
from routewise.router.replay import replay
from routewise.router.router import NoDataError, Router, load_router_config

MODE_KEYS = list(MODES)
EXAMPLES = {
    "Extract invoice fields": (
        "Extract the invoice number, issue date and total as JSON from this email: "
        "Hi, attached is invoice INV-2291 dated 3 March 2026 for $1,240.00. Thanks, Priya"
    ),
    "Find a bug": (
        "Find the bug in this function and give the line number:\n"
        "1 def average(xs):\n2     total = 0\n3     for x in xs:\n4         total += x\n"
        "5     return total / len(xs) - 1"
    ),
    "Summarize a thread": (
        "Summarize this support thread for my manager in 3 bullet points: the customer was "
        "charged twice for order #48213 ($89 each), we refunded $89 on Oct 2, and offered a "
        "15% credit."
    ),
    "Redact personal data": (
        "Redact every email address and phone number in this note: call Maria on "
        "(415) 555-0134 or email maria.lopez@northwind.example before Friday."
    ),
}


@st.cache_resource(show_spinner=False)
def load_state(db_path: str, mtime: float, models_config: str | None):
    store = ResultStore(db_path)
    tasks = all_tasks(include_studio=True)
    registry = load_models(include_disabled=True)
    rows = store.rows(current=tasks)
    return store, tasks, registry, rows, ExampleIndex.from_tasks(tasks)


@st.cache_data(show_spinner=False)
def replay_frame(db_path: str, mtime: float, models: tuple[str, ...]) -> pd.DataFrame:
    _, tasks, registry, rows, _ = load_state(
        db_path, mtime, os.environ.get("ROUTEWISE_MODELS_CONFIG")
    )
    rows = [r for r in rows if r.model_id in models]
    config = load_router_config()
    records = []
    for mode in MODE_KEYS:
        s = replay(rows, tasks, mode, config, registry)
        if not s.n_tasks:
            continue
        records.append(
            {
                "mode": mode,
                "router_cost": s.outcomes["router"].total_cost_usd,
                "flagship_cost": s.outcomes["flagship"].total_cost_usd,
                "router_quality": s.outcomes["router"].mean_quality,
                "flagship_quality": s.outcomes["flagship"].mean_quality,
                "savings": s.savings_vs_flagship,
                "quality_delta": s.quality_delta_vs_flagship,
                "tasks": s.n_tasks,
            }
        )
    return pd.DataFrame.from_records(records)


inject_css()
db_path = str(default_db_path())
mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
store, tasks, registry, rows, examples = load_state(
    db_path, mtime, os.environ.get("ROUTEWISE_MODELS_CONFIG")
)


def name(model_id: str) -> str:
    return display_name(model_id, registry)


def provider(model_id: str) -> str:
    spec = registry.get(model_id)
    return PROVIDER_LABELS.get(spec.provider, spec.provider) if spec else ""


all_models = sorted({r.model_id for r in rows}, key=lambda m: name(m))
n_tasks = len({r.task_id for r in rows})

render(
    hero(
        "Model router",
        "The right model for every prompt",
        "routewise recommends the cheapest model that is still good enough for your prompt. "
        f"Every recommendation is backed by how each model actually performed on {n_tasks} "
        "evaluation tasks, measured for quality, cost and speed.",
    )
)

with st.sidebar:
    st.markdown("**About this data**")
    st.caption(
        f"{len(rows)} evaluation results across {len(all_models)} models and {n_tasks} tasks. "
        "Add models or rerun evals with `routewise eval`."
    )

if not rows:
    render(
        callout(
            "No evaluation results yet. Run `uv run routewise import` to load the bundled "
            "results, or `uv run routewise eval --limit 4` to evaluate your own models, then "
            "refresh this page. (Full run: `uv run routewise eval`.)",
            "accent",
        )
    )
    st.stop()

frame = replay_frame(db_path, mtime, tuple(all_models))
balanced = frame[frame["mode"] == "balanced"]
saved = f"{balanced['savings'].iloc[0]:.0%}" if not balanced.empty else "n/a"
qdelta = f"{balanced['quality_delta'].iloc[0]:+.3f} quality" if not balanced.empty else ""
render(
    kpis(
        [
            ("Models compared", str(len(all_models)), ", ".join(name(m) for m in all_models)),
            ("Evaluation tasks", str(n_tasks), "across 5 task types"),
            ("Results recorded", str(len(rows)), "quality, cost and speed for each"),
            ("Cost saved vs. always using the flagship", saved, f"best-value mode, {qdelta}"),
        ]
    )
)

# --- 1. prompt --------------------------------------------------------------------------------

render(
    step(
        1,
        "Enter a prompt",
        "Paste anything you would send to an LLM. Nothing is sent to a model unless you ask "
        "for live answers below.",
    )
)


def _use_example(text: str) -> None:
    st.session_state["prompt"] = text


ex_cols = st.columns(len(EXAMPLES) + 1)
ex_cols[0].caption("Try an example:")
for col, (label, text) in zip(ex_cols[1:], EXAMPLES.items(), strict=True):
    col.button(
        label, key=f"ex-{label}", on_click=_use_example, args=(text,), use_container_width=True
    )

prompt = st.text_area(
    "Your prompt",
    key="prompt",
    height=150,
    placeholder="e.g. Extract the order number and total from this email as JSON: ...",
    label_visibility="collapsed",
)
with st.expander("Add a system prompt (optional)"):
    system = st.text_area(
        "System prompt",
        key="system",
        height=90,
        label_visibility="collapsed",
        placeholder="e.g. You are a support assistant for Acme. Never reveal internal codes.",
    )

mode = st.radio(
    "What matters most?",
    MODE_KEYS,
    index=1,
    horizontal=True,
    key="mode",
    format_func=lambda m: MODES[m][0],
    captions=[MODES[m][1] for m in MODE_KEYS],
)
with st.expander(f"Models the router may choose from ({len(all_models)} available)"):
    models = st.multiselect(
        "Models",
        all_models,
        default=all_models,
        key="models",
        format_func=name,
        label_visibility="collapsed",
    )
execute = st.checkbox(
    "Also get real answers from the recommended model and the flagship, side by side "
    "(this calls the models)",
    key="execute",
)
if st.button("Recommend a model", key="route", type="primary"):
    if not prompt.strip():
        st.warning("Enter a prompt first, or pick one of the examples above.")
        st.session_state.pop("routed", None)
    elif len(models) < 1:
        st.warning("Pick at least one model.")
    else:
        st.session_state["routed"] = {
            "prompt": prompt,
            "system": system,
            "mode": mode,
            "models": list(models),
            "execute": execute,
        }

# --- 2-4. recommendation ------------------------------------------------------------------------

routed = st.session_state.get("routed")
if routed:
    text = f"{routed['system']}\n\n{routed['prompt']}" if routed["system"] else routed["prompt"]
    board = Leaderboard.from_rows([r for r in rows if r.model_id in routed["models"]])
    router = Router(
        board, load_router_config(), registry, models=routed["models"], examples=examples
    )
    try:
        decision = router.route(text, routed["mode"])
        baseline = router.always_flagship(text)
    except NoDataError as exc:
        render(callout(str(exc)))
        st.stop()

    chosen = decision.candidates[0]
    flag = baseline.candidates[0]
    head, reason = verdict(chosen, flag, decision.category, routed["mode"], name)
    chip_items = [
        (f"Task type: {category_label(decision.category)}", "accent"),
        (f"{MODES[routed['mode']][0]} mode", ""),
    ]
    if chosen.model_id != flag.model_id and flag.mean_cost_usd:
        chip_items.append(
            (
                f"{1 - chosen.mean_cost_usd / flag.mean_cost_usd:.0%} cheaper than "
                f"{name(flag.model_id)}",
                "good",
            )
        )
    chip_items.append((f"Based on {chosen.n} past results", ""))

    render(step(2, "Recommendation", "What routewise would do with this prompt, and why."))
    render(verdict_card(head, reason, chip_items))
    blurb = CATEGORY_BLURBS.get(decision.category, "")
    st.caption(
        f"{detection_note(decision.classification)} "
        + (f"{category_label(decision.category)} covers {blurb}." if blurb else "")
    )

    used = category_label(decision.used_category)
    render(
        step(
            3,
            f"How every model compares on {category_label(decision.category).lower()} tasks",
            f"Averages from past evaluation runs ({used.lower()}). These are what to expect "
            "from each model on tasks like yours, not measurements of this exact prompt.",
        )
    )
    max_cost = max(c.mean_cost_usd for c in decision.candidates) or 1.0
    ordered = decision.candidates
    per_row = 3
    for start in range(0, len(ordered), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, c in zip(cols, ordered[start : start + per_row], strict=False):
            if c.model_id == chosen.model_id:
                badge, badge_kind, kind = "Recommended", "accent", "chosen"
            elif not c.eligible:
                badge, badge_kind, kind = "Not eligible", "warn", "dim"
            elif c.model_id == flag.model_id:
                badge, badge_kind, kind = "Flagship", "", ""
            else:
                badge, badge_kind, kind = "Considered", "", ""
            if c.model_id == flag.model_id and badge == "Recommended":
                badge = "Recommended · Flagship"
            foot = f"Based on {c.n} past results."
            if not c.eligible:
                foot += f" Ruled out: {c.excluded_reason}."
            col.markdown(
                model_card(
                    name=name(c.model_id),
                    provider=provider(c.model_id),
                    badge=badge,
                    badge_kind=badge_kind,
                    kind=kind,
                    stats=[
                        ("Expected quality", f"{c.quality:.2f} / 1.00", c.quality),
                        ("Cost per query", money(c.mean_cost_usd), c.mean_cost_usd / max_cost),
                        ("Typical response time", f"{c.mean_latency_s:.1f}s", None),
                    ],
                    footnote=foot,
                ),
                unsafe_allow_html=True,
            )
    with st.expander("Full reasoning"):
        st.write(escape_markdown(decision.rationale))

    if routed["execute"]:
        render(
            step(
                4,
                "Real answers, side by side",
                "The recommended model and the flagship, run on your exact prompt.",
            )
        )
        key = (text, routed["mode"], tuple(routed["models"]))
        cache = st.session_state.setdefault("live", {})
        if key not in cache:
            messages = [{"role": "system", "content": routed["system"]}] if routed["system"] else []
            messages.append({"role": "user", "content": routed["prompt"]})
            chat_fn = functools.partial(chat, registry=registry)
            with st.spinner("Calling the models..."):
                try:
                    _, routed_result, _ = asyncio.run(
                        router.complete(messages, routed["mode"], chat_fn=chat_fn)
                    )
                    flag_result = (
                        routed_result
                        if flag.model_id == routed_result.model_id
                        else asyncio.run(chat_fn(flag.model_id, messages))
                    )
                    cache[key] = (routed_result, flag_result)
                except Exception as exc:  # show the failure, keep the page usable
                    render(callout(f"Live call failed: {type(exc).__name__}: {exc}"))
        if key in cache:
            routed_result, flag_result = cache[key]
            pairs = [("Recommended", routed_result)]
            if flag_result.model_id != routed_result.model_id:
                pairs.append(("Flagship", flag_result))
            cols = st.columns(len(pairs), gap="medium")
            for col, (label, res) in zip(cols, pairs, strict=True):
                with col, st.container(border=True):
                    st.markdown(f"**{label}: {name(res.model_id)}**")
                    st.caption(
                        f"{res.latency_s:.1f}s · {escape_markdown(money(res.cost_usd))}"
                        " for this answer"
                    )
                    with st.container(height=380, border=False):
                        st.markdown(demote_headings(escape_markdown(res.text)))

# --- evidence ---------------------------------------------------------------------------------

render(
    step(
        "★",
        "The evidence behind every recommendation",
        "Measured, not assumed: every model was run on the same evaluation tasks.",
    )
)
tab_quality, tab_savings, tab_how = st.tabs(
    ["Quality by task type", "Savings vs always using the flagship", "How routing works"]
)

with tab_quality:
    board_all = Leaderboard.from_rows(rows)
    recs = pd.DataFrame(board_all.records())
    recs["Model"] = recs["model_id"].map(name)
    recs["Task type"] = recs["category"].map(category_label)
    order = [category_label(c) for c in board_all.categories()] + [category_label(OVERALL)]
    q_low = max(0.0, float(recs["mean_quality"].min()) - 0.02)
    mid = (q_low + 1.0) / 2
    base = alt.Chart(recs).encode(
        x=alt.X(
            "Task type:N",
            sort=order,
            title=None,
            axis=alt.Axis(
                labelAngle=0,
                orient="top",
                labelFontSize=12,
                ticks=False,
                domain=False,
                labelLimit=220,
                labelColor="#52514e",
            ),
        ),
        y=alt.Y(
            "Model:N",
            title=None,
            axis=alt.Axis(
                labelFontSize=13, ticks=False, domain=False, labelLimit=220, labelColor="#0b0b0b"
            ),
        ),
    )
    heat = base.mark_rect(cornerRadius=4, stroke="#fcfcfb", strokeWidth=2).encode(
        color=alt.Color(
            "mean_quality:Q",
            title="Quality",
            scale=alt.Scale(domain=[q_low, 1], range=SEQ_RAMP[:6]),
            legend=alt.Legend(orient="right", gradientLength=120),
        ),
        tooltip=[
            alt.Tooltip("Model:N"),
            alt.Tooltip("Task type:N"),
            alt.Tooltip("mean_quality:Q", title="Mean quality", format=".3f"),
            alt.Tooltip("ci_low:Q", title="95% CI low", format=".2f"),
            alt.Tooltip("ci_high:Q", title="95% CI high", format=".2f"),
            alt.Tooltip("n:Q", title="Results"),
            alt.Tooltip("mean_cost_usd:Q", title="Mean cost ($)", format=".5f"),
        ],
    )
    labels = base.mark_text(fontSize=13, fontWeight=600).encode(
        text=alt.Text("mean_quality:Q", format=".2f"),
        color=alt.condition(
            f"datum.mean_quality > {mid}", alt.value("white"), alt.value("#0b0b0b")
        ),
    )
    st.altair_chart((heat + labels).properties(height=alt.Step(46)), width="stretch")
    st.caption(
        "Each cell is a model's average quality score (0 to 1) on one task type. Hover for the "
        "95% confidence interval, number of results and cost. Darker means better; the colour "
        "scale starts just below the lowest score so small differences stay visible."
    )
    with st.expander("Show as a table"):
        st.dataframe(
            recs[
                [
                    "Model",
                    "Task type",
                    "mean_quality",
                    "ci_low",
                    "ci_high",
                    "n",
                    "mean_cost_usd",
                    "mean_latency_s",
                ]
            ].rename(
                columns={
                    "mean_quality": "Quality",
                    "ci_low": "CI low",
                    "ci_high": "CI high",
                    "n": "Results",
                    "mean_cost_usd": "Mean cost ($)",
                    "mean_latency_s": "Mean latency (s)",
                }
            ),
            hide_index=True,
            width="stretch",
        )

with tab_savings:
    if frame.empty:
        render(callout("Not enough overlapping results across models to replay."))
    else:
        tiles = []
        for _, r in frame.iterrows():
            tiles.append(
                (
                    f"{MODES[r['mode']][0]} mode",
                    f"{r['savings']:.0%} cheaper",
                    f"quality {r['router_quality']:.3f} vs {r['flagship_quality']:.3f} flagship",
                )
            )
        render(kpis(tiles))
        long = pd.concat(
            [
                frame.assign(policy="routewise", cost=frame["router_cost"]),
                frame.assign(policy="Always the flagship", cost=frame["flagship_cost"]),
            ]
        )
        long["Mode"] = long["mode"].map(lambda m: MODES[m][0])
        bars = (
            alt.Chart(long)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=34)
            .encode(
                x=alt.X(
                    "Mode:N",
                    sort=[MODES[m][0] for m in MODE_KEYS],
                    title=None,
                    axis=alt.Axis(labelAngle=0, labelFontSize=12),
                ),
                xOffset=alt.XOffset("policy:N", sort=["Always the flagship", "routewise"]),
                y=alt.Y(
                    "cost:Q",
                    title=f"Total cost over {int(frame['tasks'].iloc[0])} tasks ($)",
                    axis=alt.Axis(grid=True, gridColor="#eeede9", domain=False),
                ),
                color=alt.Color(
                    "policy:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Always the flagship", "routewise"], range=[GRAY_MARK, ACCENT]
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    "Mode",
                    "policy",
                    alt.Tooltip("cost:Q", title="Total cost ($)", format=".4f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(bars, width="stretch")
        st.caption(
            "Leave-one-out replay: each evaluation task is routed by a router that has never "
            "seen that task's results, then charged at the recorded cost of the model it chose. "
            "Costs are list-price equivalents."
        )

with tab_how:
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown(
        model_card(
            name="1. Recognise the task",
            provider="",
            badge="Classify",
            stats=[],
            footnote="Your prompt (and system prompt) is matched to a task type: structured "
            "output, reasoning, summarization, voice or safety. It compares against the "
            "labelled evaluation tasks first, then keyword rules.",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        model_card(
            name="2. Look up the evidence",
            provider="",
            badge="Measure",
            stats=[],
            footnote="For that task type, routewise looks up every model's measured average "
            "quality, cost per query and response time from the evaluation runs.",
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        model_card(
            name="3. Score the trade-off",
            provider="",
            badge="Decide",
            stats=[],
            footnote="Each model scores quality minus a cost penalty minus a speed penalty. The "
            "mode sets the penalties, and Best value and Lowest cost refuse models too "
            "far below the best quality. Highest score wins.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.dataframe(
        pd.DataFrame([{"Mode": MODES[m][0], "What it optimises": MODES[m][1]} for m in MODE_KEYS]),
        hide_index=True,
        width="stretch",
    )
