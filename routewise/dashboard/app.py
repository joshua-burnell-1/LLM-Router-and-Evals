"""routewise dashboard.

    uv run routewise dashboard      (or: uv run streamlit run routewise/dashboard/app.py)

Reads eval results from data/results.db (or $ROUTEWISE_DB).
"""

from __future__ import annotations

import asyncio
import functools
import os

import altair as alt
import pandas as pd
import streamlit as st

from routewise.config import default_db_path
from routewise.dashboard import escape_markdown
from routewise.evals.loader import all_tasks
from routewise.evals.store import ResultStore
from routewise.models.client import chat
from routewise.models.registry import load_models
from routewise.router.classifier import ExampleIndex
from routewise.router.leaderboard import OVERALL, Leaderboard
from routewise.router.replay import replay
from routewise.router.router import NoDataError, Router, load_router_config

MODES = ["quality", "balanced", "cost"]

st.set_page_config(page_title="routewise", layout="wide")


@st.cache_resource(show_spinner=False)
def load_state(db_path: str, mtime: float, models_config: str | None):
    """Everything derived from the results DB; the cache key changes when the DB does."""
    store = ResultStore(db_path)
    tasks = all_tasks()
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
    for mode in MODES:
        s = replay(rows, tasks, mode, config, registry)
        if not s.n_tasks:
            continue
        for policy in ("router", "flagship"):
            o = s.outcomes[policy]
            records.append(
                {
                    "mode": mode,
                    "policy": "router" if policy == "router" else "always flagship",
                    "total_cost_usd": o.total_cost_usd,
                    "mean_quality": o.mean_quality,
                    "savings": s.savings_vs_flagship if policy == "router" else 0.0,
                    "quality_delta": s.quality_delta_vs_flagship if policy == "router" else 0.0,
                    "tasks": s.n_tasks,
                }
            )
    return pd.DataFrame.from_records(records)


db_path = str(default_db_path())
mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
store, tasks, registry, rows, examples = load_state(
    db_path, mtime, os.environ.get("ROUTEWISE_MODELS_CONFIG")
)

st.title("routewise")
st.caption(
    "Eval-gated LLM routing: every decision is backed by measured quality, cost and latency."
)

all_models = sorted({r.model_id for r in rows})
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Routing mode", MODES, index=1)
    models = st.multiselect("Models", all_models, default=all_models)
    execute = st.checkbox("Execute query live (calls the models)", value=False)
    st.caption(f"Results DB: `{db_path}`  \n{len(rows)} result rows, {len(tasks)} tasks")

if not rows:
    st.info(
        "No eval results yet. Run `uv run routewise eval --limit 4` to validate the pipeline, "
        "then `uv run routewise eval` for the full suite."
    )
    st.stop()
if not models:
    st.warning("Select at least one model.")
    st.stop()

board = Leaderboard.from_rows([r for r in rows if r.model_id in models])
router = Router(board, load_router_config(), registry, models=models, examples=examples)

# --- route a query -------------------------------------------------------------------------

st.subheader("Route a query")
with st.form("route"):
    query = st.text_area("Query", placeholder="e.g. Extract the order number and amount as JSON")
    submitted = st.form_submit_button("Route")

if submitted and query.strip():
    try:
        decision = router.route(query, mode)
        baseline = router.always_flagship(query)
    except NoDataError as exc:
        st.error(str(exc))
    else:
        routed = decision.candidates[0]
        flag = baseline.candidates[0]  # flagship stats (overall if it lacks category data)
        c1, c2 = st.columns(2)
        c1.metric("Routed model", decision.model_id)
        c2.metric("Flagship baseline", baseline.model_id)
        st.info(escape_markdown(decision.rationale))

        left, right = st.columns(2)
        with left:
            st.markdown(f"**Routed: {routed.model_id}**")
            a, b, c = st.columns(3)
            a.metric("Quality", f"{routed.quality:.2f}", f"{routed.quality - flag.quality:+.2f}")
            b.metric(
                "Latency",
                f"{routed.mean_latency_s:.1f}s",
                f"{routed.mean_latency_s - flag.mean_latency_s:+.1f}s",
                delta_color="inverse",
            )
            c.metric(
                "Cost / query",
                f"${routed.mean_cost_usd:.5f}",
                f"{(routed.mean_cost_usd / flag.mean_cost_usd - 1):+.0%}"
                if flag.mean_cost_usd
                else None,
                delta_color="inverse",
            )
        with right:
            st.markdown(f"**Flagship: {flag.model_id}**")
            a, b, c = st.columns(3)
            a.metric("Quality", f"{flag.quality:.2f}")
            b.metric("Latency", f"{flag.mean_latency_s:.1f}s")
            c.metric("Cost / query", f"${flag.mean_cost_usd:.5f}")
        st.caption(
            f"Expected values are {decision.used_category} means from the eval set "
            f"(category {decision.category}, classified by {decision.classification.method})."
        )

        ranking = pd.DataFrame(
            [
                {
                    "model": c.model_id,
                    "score": round(c.score, 3),
                    "quality": round(c.quality, 3),
                    "mean cost ($)": c.mean_cost_usd,
                    "mean latency (s)": round(c.mean_latency_s, 2),
                    "n": c.n,
                    "eligible": c.eligible,
                    "excluded because": c.excluded_reason,
                }
                for c in decision.candidates
            ]
        )
        st.dataframe(ranking, hide_index=True, width="stretch")

        if execute:
            chat_fn = functools.partial(chat, registry=registry)
            with st.spinner("Calling models..."):
                _, result, attempts = asyncio.run(router.complete(query, mode, chat_fn=chat_fn))
                flag_result = asyncio.run(
                    chat_fn(baseline.model_id, [{"role": "user", "content": query}])
                )
            out_l, out_r = st.columns(2)
            out_l.markdown(
                escape_markdown(
                    f"**{result.model_id}** ({result.latency_s:.1f}s, ${result.cost_usd:.5f})"
                )
            )
            out_l.write(result.text)
            out_r.markdown(
                escape_markdown(
                    f"**{flag_result.model_id}** ({flag_result.latency_s:.1f}s, "
                    f"${flag_result.cost_usd:.5f})"
                )
            )
            out_r.write(flag_result.text)

# --- leaderboard ------------------------------------------------------------------------------

st.subheader("Leaderboard: mean quality by category")
records = pd.DataFrame(board.records())
# Scores cluster near 1.0, so the colour scale starts just below the lowest observed value
# (the legend shows the range); a fixed 0-1 scale would paint every cell the same blue.
q_low = max(0.0, float(records["mean_quality"].min()) - 0.02)
q_mid = (q_low + 1.0) / 2
heat = (
    alt.Chart(records)
    .mark_rect()
    .encode(
        x=alt.X(
            "category:N",
            title=None,
            sort=board.categories() + [OVERALL],
            axis=alt.Axis(labelAngle=0, orient="top"),
        ),
        y=alt.Y("model_id:N", title=None),
        color=alt.Color(
            "mean_quality:Q",
            scale=alt.Scale(scheme="blues", domain=[q_low, 1]),
            title="quality",
        ),
        tooltip=[
            "model_id",
            "category",
            alt.Tooltip("mean_quality:Q", format=".3f"),
            alt.Tooltip("ci_low:Q", format=".2f"),
            alt.Tooltip("ci_high:Q", format=".2f"),
            "n",
            alt.Tooltip("mean_cost_usd:Q", format=".5f"),
            alt.Tooltip("mean_latency_s:Q", format=".2f"),
        ],
    )
    .properties(height=alt.Step(44))
)
labels = heat.mark_text(baseline="middle", fontSize=13).encode(
    text=alt.Text("mean_quality:Q", format=".2f"),
    color=alt.condition(f"datum.mean_quality > {q_mid}", alt.value("white"), alt.value("black")),
)
st.altair_chart(heat + labels, width="stretch")

overall = records[records["category"] == OVERALL][
    [
        "model_id",
        "mean_quality",
        "ci_low",
        "ci_high",
        "mean_latency_s",
        "mean_cost_usd",
        "total_cost_usd",
        "n",
    ]
].sort_values("mean_quality", ascending=False)
st.dataframe(overall, hide_index=True, width="stretch")

# --- savings ------------------------------------------------------------------------------------

st.subheader("Cost savings: router vs always-flagship (leave-one-out replay of the eval set)")
frame = replay_frame(db_path, mtime, tuple(models))
if frame.empty:
    st.info("Not enough overlapping results across the selected models to replay.")
else:
    cols = st.columns(len(MODES))
    for col, m in zip(cols, MODES, strict=False):
        row = frame[(frame["mode"] == m) & (frame["policy"] == "router")]
        if not row.empty:
            col.metric(
                f"{m} mode savings",
                f"{row['savings'].iloc[0]:.0%}",
                f"{row['quality_delta'].iloc[0]:+.3f} quality vs flagship",
                delta_color="off",
            )
    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("policy:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("total_cost_usd:Q", title="total cost over eval set ($)"),
            color=alt.Color("policy:N", legend=None),
            column=alt.Column("mode:N", title=None, sort=MODES),
            tooltip=[
                "mode",
                "policy",
                alt.Tooltip("total_cost_usd:Q", format=".4f"),
                alt.Tooltip("mean_quality:Q", format=".3f"),
                "tasks",
            ],
        )
        .properties(width=140)
    )
    st.altair_chart(bars)
    st.caption(
        "Each eval task is routed with a leaderboard that excludes that task, then scored with "
        "the recorded result of the chosen model. Costs are list-price equivalents."
    )
