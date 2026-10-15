"""Benchmark studio: benchmark any model on your own prompt, scored by the checks you choose."""

from __future__ import annotations

import asyncio
import functools
import io
import os

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from routewise.config import default_db_path
from routewise.dashboard import demote_headings, escape_markdown
from routewise.dashboard.narrative import CATEGORY_BLURBS, category_label
from routewise.dashboard.ui import (
    SEQ_RAMP,
    callout,
    chips,
    hero,
    inject_css,
    kpis,
    model_card,
    money,
    render,
    step,
    verdict_card,
)
from routewise.evals.framework import CATEGORIES, LLMJudge
from routewise.evals.store import ResultStore
from routewise.models.client import chat
from routewise.models.registry import PROVIDER_LABELS, display_name, load_models
from routewise.router.classifier import classify
from routewise.settings import model_status
from routewise.studio.catalog import CATALOG, CHECKS, defaults
from routewise.studio.model import CheckConfig, StudioBenchmark, list_saved, save
from routewise.studio.report import html_report
from routewise.studio.run import estimate_cost, run_studio, summarize, verdict
from routewise.studio.templates import TEMPLATES

inject_css()
registry = load_models(include_disabled=True)
store = ResultStore(default_db_path())
ready = [m for m, spec in registry.items() if model_status(spec)[0] == "ready"]
ss = st.session_state


def name(model_id: str) -> str:
    return display_name(model_id, registry)


def provider(model_id: str) -> str:
    spec = registry.get(model_id)
    return PROVIDER_LABELS.get(spec.provider, spec.provider) if spec else ""


# --- loading a benchmark into the form ----------------------------------------------------------


def _load(bench: StudioBenchmark) -> None:
    ss["st-name"] = bench.name
    ss["st-description"] = bench.description
    ss["st-prompt"] = bench.prompt
    ss["st-system"] = bench.system_prompt or ""
    ss["st-category"] = bench.category
    ss["st-checks"] = [c.key for c in bench.checks if c.key in CATALOG]
    for info in CHECKS:
        base = defaults(info.key)
        chosen = next((c for c in bench.checks if c.key == info.key), None)
        values = {**base, **(chosen.values if chosen else {})}
        for p in info.params:
            ss[f"chk-{info.key}-{p.name}"] = values.get(p.name)
        ss[f"chk-{info.key}-weight"] = float(chosen.weight) if chosen else 1.0
        ss[f"chk-{info.key}-required"] = bool(chosen.required) if chosen else False
    if bench.judge_model and bench.judge_model in registry:
        ss["st-judge"] = bench.judge_model
    ss["st-cases-base"] = [dict(c) for c in bench.cases]
    ss["st-cases-ver"] = ss.get("st-cases-ver", 0) + 1
    ss.pop("st-cases-cols", None)
    ss.pop("st-cases-last", None)  # the loaded cases win over whatever was being edited
    ss.pop("st-last-run", None)


if "studio-load" in ss:
    _load(ss.pop("studio-load"))
if "st-prompt" not in ss:
    _load(TEMPLATES[0])

load_errors: list[str] = []
saved = {b.slug: b for b in list_saved(errors=load_errors)}

render(
    hero(
        "Benchmark studio",
        "Benchmark any model on your own work",
        "Write the task once, add a few test cases, choose the checks that define a good "
        "answer, and run it across models. You get a clear winner on quality and on value, every "
        "answer side by side, and a shareable report. Results also teach the router.",
    )
)

# --- start from -----------------------------------------------------------------------------

options = [f"template:{i}" for i in range(len(TEMPLATES))] + [f"saved:{s}" for s in saved]


def _option_label(opt: str) -> str:
    kind, ref = opt.split(":", 1)
    if kind == "template":
        t = TEMPLATES[int(ref)]
        return f"Template · {t.name}: {t.description}"
    return f"Saved · {saved[ref].name}"


if load_errors:
    st.warning("Some saved benchmarks could not be read and are hidden: " + "; ".join(load_errors))
c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
start = c1.selectbox("Start from", options, format_func=_option_label, key="st-start")
if c2.button("Load", key="st-load", use_container_width=True):
    kind, ref = start.split(":", 1)
    ss["studio-load"] = TEMPLATES[int(ref)] if kind == "template" else saved[ref]
    st.rerun()

# --- 1. describe the task -----------------------------------------------------------------------

render(
    step(
        1,
        "Describe the task",
        "Write the prompt exactly as you would send it. Use {{name}} for anything that "
        "changes between test cases, for example {{email}}.",
    )
)
left, right = st.columns([2, 1], gap="large")
with left:
    bench_name = st.text_input("Benchmark name", key="st-name")
    prompt = st.text_area("Prompt template", key="st-prompt", height=170)
    with st.expander("System prompt (optional)"):
        system = st.text_area(
            "System prompt", key="st-system", height=100, label_visibility="collapsed"
        )
with right:
    description = st.text_input("One-line description (optional)", key="st-description")
    category = st.selectbox(
        "Task type",
        list(CATEGORIES),
        key="st-category",
        format_func=category_label,
        help="Results feed the router's evidence for this type of task.",
    )
    st.caption(CATEGORY_BLURBS.get(category, "").capitalize() + ".")
    detected = classify(f"{system}\n\n{prompt}").category
    if detected in CATEGORIES and detected != category:
        st.caption(f"This prompt reads like **{category_label(detected).lower()}**.")

draft = StudioBenchmark(
    name=bench_name, category=category, prompt=prompt, system_prompt=system or None
)
variables = draft.variables()
if variables:
    render(chips([(f"{{{{{v}}}}}", "accent") for v in variables]))

# --- 2. test cases ----------------------------------------------------------------------------

selected_checks = ss.get("st-checks", [])
needs_expected = any(CATALOG[k].uses_expected for k in selected_checks if k in CATALOG) or (
    "llm_judge" in selected_checks and bool(ss.get("chk-llm_judge-show_expected"))
)
columns = list(variables) + (["expected"] if needs_expected and "expected" not in variables else [])

render(
    step(
        2,
        "Add test cases",
        "One row per test case. Fill a value for every variable"
        + (" and the expected answer." if needs_expected else ".")
        + " Add rows with the + under the table, or upload a CSV.",
    )
)
if ss.get("st-cases-cols") != columns:
    previous = ss.get("st-cases-last") or ss.get("st-cases-base") or []
    ss["st-cases-base"] = [{c: str(r.get(c, "") or "") for c in columns} for r in previous] or [
        {c: "" for c in columns}
    ]
    ss["st-cases-cols"] = columns
    ss["st-cases-ver"] = ss.get("st-cases-ver", 0) + 1

upload = st.file_uploader("Upload test cases (CSV with a header row)", type=["csv"], key="st-csv")
if upload is None:
    ss.pop("st-csv-seen", None)  # so re-uploading the same file after a fix is picked up
elif ss.get("st-csv-seen") != (upload.name, upload.size):
    ss["st-csv-seen"] = (upload.name, upload.size)
    frame = None
    for encoding in ("utf-8-sig", "cp1252"):  # Excel on Windows saves cp1252
        try:
            frame = pd.read_csv(io.BytesIO(upload.getvalue()), dtype=str, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            st.error(f"Could not read that CSV: {exc}")
            break
    if frame is not None:
        frame = frame.fillna("")
        frame.columns = [str(c).strip() for c in frame.columns]
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            st.error("The CSV is missing columns: " + ", ".join(missing))
        else:
            ss["st-cases-base"] = frame[columns].to_dict("records")
            ss["st-cases-ver"] = ss.get("st-cases-ver", 0) + 1

if columns:
    edited = st.data_editor(
        pd.DataFrame(ss["st-cases-base"], columns=columns),
        key=f"st-cases-{ss['st-cases-ver']}",
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={c: st.column_config.TextColumn(c, width="large") for c in columns},
    )
    cases = [
        {c: str(v) for c, v in row.items()}
        for row in edited.fillna("").to_dict("records")
        if any(str(v).strip() for v in row.values())
    ]
else:
    st.caption("This prompt has no variables, so it runs once per model.")
    cases = []
ss["st-cases-last"] = cases

# --- 3. models ----------------------------------------------------------------------------------

render(
    step(
        3,
        "Choose the models",
        "Every model answers every test case. Include the model you use today to compare "
        "against it.",
    )
)
if len(ready) < 1:
    render(
        callout(
            "No models are ready. Open Models & keys in the sidebar to add an API key.", "accent"
        )
    )
models = st.multiselect(
    "Models",
    ready,
    default=ss.get("st-models-default", ready[:3]),
    key="st-models",
    format_func=name,
)
not_ready = [m for m, spec in registry.items() if spec.enabled and m not in ready]
if not_ready:
    st.caption(
        "Not available yet: "
        + ", ".join(name(m) for m in not_ready)
        + ". Add keys in Models & keys."
    )

# --- 4. checks ----------------------------------------------------------------------------------

render(
    step(
        4,
        "Choose how answers are scored",
        "Each check gives a score from 0 to 1; an answer's score is the weighted average. "
        "Mark a check as must-pass to fail the whole answer when it scores 0.",
    )
)
st.multiselect(
    "Checks", [c.key for c in CHECKS], key="st-checks", format_func=lambda k: CATALOG[k].label
)
check_configs: list[CheckConfig] = []
judge_model = None
for key in ss.get("st-checks", []):
    info = CATALOG[key]
    with st.container(border=True):
        st.markdown(f"**{info.label}**")
        st.caption(info.description)
        values = {}
        cols = st.columns(max(1, len(info.params)) + 2, vertical_alignment="bottom")
        for col, p in zip(cols, info.params, strict=False):
            wkey = f"chk-{key}-{p.name}"
            if wkey not in ss:
                ss[wkey] = p.default
            if p.kind == "textarea":
                values[p.name] = col.text_area(p.label, key=wkey, help=p.help or None, height=140)
            elif p.kind == "int":
                values[p.name] = col.number_input(
                    p.label, key=wkey, min_value=0, step=5, help=p.help or None
                )
            elif p.kind == "choice":
                values[p.name] = col.selectbox(
                    p.label, list(p.choices), key=wkey, help=p.help or None
                )
            elif p.kind == "bool":
                values[p.name] = col.checkbox(p.label, key=wkey, help=p.help or None)
            elif p.kind == "column":
                if ss.get(wkey) not in variables:
                    ss[wkey] = variables[0] if variables else None
                values[p.name] = col.selectbox(
                    p.label, variables or ["(no variables)"], key=wkey, help=p.help or None
                )
            else:
                values[p.name] = col.text_input(p.label, key=wkey, help=p.help or None)
        wcol, rcol = cols[-2], cols[-1]
        wkey, rkey = f"chk-{key}-weight", f"chk-{key}-required"
        ss.setdefault(wkey, 1.0)
        ss.setdefault(rkey, False)
        weight = wcol.number_input("Weight", key=wkey, min_value=0.5, max_value=5.0, step=0.5)
        required = rcol.checkbox("Must pass", key=rkey)
        if key == "llm_judge":
            judge_options = ready or list(registry)
            default_judge = (
                "claude-haiku-5-5"
                if "claude-haiku-5-5" in judge_options
                else (judge_options[0] if judge_options else None)
            )
            if ss.get("st-judge") not in judge_options:
                ss["st-judge"] = default_judge
            judge_model = st.selectbox(
                "Judge model",
                judge_options,
                key="st-judge",
                format_func=name,
                help="A cheap, consistent model is usually best.",
            )
        check_configs.append(CheckConfig(key, values, float(weight), bool(required)))

bench = StudioBenchmark(
    name=bench_name,
    category=category,
    prompt=prompt,
    cases=cases,
    checks=check_configs,
    system_prompt=system or None,
    description=description,
    judge_model=judge_model or "",
)

# --- 5. review and run ------------------------------------------------------------------------

render(step(5, "Review and run"))
problems = bench.validate()
if not models:
    problems.append("Pick at least one model.")
if problems:
    render(callout("Before you can run: " + " ".join(problems)))
    st.markdown("")
estimate = (
    estimate_cost(bench, models, registry, judge_model=judge_model, store=store)
    if not problems
    else None
)
if estimate:
    render(
        kpis(
            [
                (
                    "New model calls",
                    str(estimate.calls),
                    f"{len(bench.effective_cases())} test cases × {len(models)} models"
                    + (f", {estimate.cached} already answered (free)" if estimate.cached else ""),
                ),
                (
                    "Estimated cost",
                    money(estimate.total_usd),
                    f"list prices, about {estimate.assumed_output_tokens} output tokens per answer",
                ),
                (
                    "Judge cost",
                    money(estimate.judge_usd) if bench.uses_judge() else "none",
                    (
                        f"{estimate.judge_calls} verdicts from {name(judge_model)}"
                        if bench.uses_judge() and judge_model
                        else "no LLM judge"
                    ),
                ),
            ]
        )
    )
c1, c2, c3 = st.columns([1.2, 1, 1], vertical_alignment="bottom")
budget = c1.number_input(
    "Stop if spend reaches ($)", min_value=0.05, value=2.0, step=0.5, key="st-budget"
)
save_clicked = c2.button(
    "Save benchmark", key="st-save", use_container_width=True, disabled=not bench.name.strip()
)
run_clicked = c3.button(
    "Run benchmark", key="st-run", type="primary", use_container_width=True, disabled=bool(problems)
)
if save_clicked:
    try:
        path = save(bench)
    except ValueError as exc:
        st.error(f"Not saved: {exc}.")
    else:
        ss["st-flash"] = (bench.name, str(path))
        st.rerun()  # so the "Start from" list picks up the new benchmark
if "st-flash" in ss:
    flash_name, flash_path = ss.pop("st-flash")
    st.success(f"Saved “{flash_name}”. Rerun it any time, for example when a new model ships.")
    st.caption(f"Stored at {flash_path} (gitignored).")
if run_clicked:
    try:
        save(bench)
    except ValueError as exc:
        st.error(f"Not run: {exc}.")
        run_clicked = False
if run_clicked:
    chat_fn = functools.partial(chat, registry=registry)
    judge = LLMJudge(judge_model, chat_fn=chat_fn) if bench.uses_judge() and judge_model else None
    with st.status(f"Running {estimate.calls} new answers...", expanded=False) as status:
        summary = asyncio.run(
            run_studio(
                bench,
                models,
                store,
                chat_fn=chat_fn,
                judge=judge,
                progress=lambda line: status.write(line),
                max_cost=budget,
            )
        )
        label = f"{summary.n_done} answered, {summary.n_skipped} reused"
        if summary.n_errors:
            label += f", {summary.n_errors} failed"
        status.update(label=label, state="error" if summary.n_errors else "complete")
    if summary.stopped_for_budget:
        st.warning(
            "Stopped at your budget cap. Raise it and run again to finish; finished "
            "answers are kept."
        )
    if summary.stopped_for_quota:
        st.warning("A provider reported the plan or credit limit is used up. Run again later.")
    ss["st-last-run"] = bench.slug

# --- results ------------------------------------------------------------------------------------

if problems or not models:
    st.stop()
results = summarize(bench, models, store, registry)
if not results.models:
    if ss.get("st-last-run") == bench.slug:
        render(callout("No results were recorded. Check the run log above for errors."))
    st.stop()

render(
    step(
        "★",
        "Results",
        f"{results.total_cases} test cases × {len(results.models)} models. "
        "Scores are 0 to 1, averaged over your checks.",
    )
)
head, line = verdict(results, name)
tied = results.tied_best()
if results.pending_judge:
    render(
        callout(
            f"{results.pending_judge} answers have no AI-judge verdict for the current rubric "
            "yet, so scores below leave that check out. Press Run benchmark to judge them "
            "(only the judge is called; stored answers are reused).",
            "accent",
        )
    )
render(
    verdict_card(
        head,
        line,
        [
            (f"Task type: {category_label(bench.category)}", "accent"),
            (f"{results.total_cases} test cases", ""),
            ("These results now feed the Model router", "good"),
        ],
    )
)
st.markdown("")

max_cost = max(m.mean_cost_usd for m in results.models) or 1.0
cols = st.columns(min(3, len(results.models)), gap="medium")
for i, m in enumerate(results.models):
    if m.model_id == results.best_value:
        badge, bkind, kind = "Best value", "accent", "chosen"
    elif m.model_id == results.best_quality and len(tied) == 1:
        badge, bkind, kind = "Best quality", "accent", ""
    elif m.model_id in tied:
        badge, bkind, kind = "Top score (tie)", "", ""
    else:
        badge, bkind, kind = f"#{i + 1}", "", ""
    foot = f"{m.cases} of {results.total_cases} test cases answered."
    if m.errors:
        foot += f" {m.errors} failed."
    cols[i % len(cols)].markdown(
        model_card(
            name=name(m.model_id),
            provider=provider(m.model_id),
            badge=badge,
            badge_kind=bkind,
            kind=kind,
            stats=[
                ("Average score", f"{m.mean_score:.2f}", m.mean_score),
                (
                    f"Pass rate (score ≥ {results.pass_threshold:.1f})",
                    f"{m.pass_rate:.0%}",
                    m.pass_rate,
                ),
                ("Cost per answer", money(m.mean_cost_usd), m.mean_cost_usd / max_cost),
                ("Typical response time", f"{m.mean_latency_s:.1f}s", None),
            ],
            footnote=foot,
        ),
        unsafe_allow_html=True,
    )

check_rows = [
    {"Model": name(m.model_id), "Check": label, "Score": v}
    for m in results.models
    for label, v in m.checks.items()
]
if check_rows:
    render(step("◆", "Score by check", "Where each model gains or loses points."))
    frame = pd.DataFrame(check_rows)
    grid = alt.Chart(frame).encode(
        x=alt.X(
            "Model:N",
            sort=[name(m.model_id) for m in results.models],
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
            "Check:N",
            title=None,
            axis=alt.Axis(labelFontSize=12, ticks=False, domain=False, labelLimit=300),
        ),
    )
    heat = grid.mark_rect(cornerRadius=4, stroke="#fcfcfb", strokeWidth=2).encode(
        color=alt.Color(
            "Score:Q",
            scale=alt.Scale(domain=[0, 1], range=SEQ_RAMP[:6]),
            legend=alt.Legend(orient="right", gradientLength=110, values=[0, 0.5, 1]),
        ),
        tooltip=["Model", "Check", alt.Tooltip("Score:Q", format=".2f")],
    )
    text = grid.mark_text(fontSize=13, fontWeight=600).encode(
        text=alt.Text("Score:Q", format=".2f"),
        color=alt.condition("datum.Score > 0.55", alt.value("white"), alt.value("#0b0b0b")),
    )
    st.altair_chart((heat + text).properties(height=alt.Step(42)), width="stretch")

render(
    step(
        "◆",
        "Every answer",
        "Pick a test case to read each model's answer and why it scored the way it did.",
    )
)
by_case: dict[int, list] = {}
for c in results.cases:
    by_case.setdefault(c.case_index, []).append(c)
case_values = bench.effective_cases()


def _case_label(i: int) -> str:
    vals = case_values[i] if i < len(case_values) else {}
    preview = " · ".join(str(v)[:50] for k, v in vals.items() if k != "expected")
    return f"Test case {i + 1}: {preview}" if preview else f"Test case {i + 1}"


idx = st.selectbox("Test case", sorted(by_case), format_func=_case_label, key="st-case-pick")
if idx is not None:
    vals = case_values[idx] if idx < len(case_values) else {}
    if vals.get("expected"):
        st.caption(f"Expected: {vals['expected']}")
    rank = {m.model_id: i for i, m in enumerate(results.models)}
    answers = sorted(
        by_case[idx],
        key=lambda c: (-(c.score if c.score is not None else -1), rank.get(c.model_id, 99)),
    )
    cols = st.columns(min(3, len(answers)), gap="medium")
    for i, c in enumerate(answers):
        with cols[i % len(cols)], st.container(border=True):
            score = "failed" if c.score is None else f"{c.score:.2f}"
            st.markdown(f"**{name(c.model_id)}** · score {score}")
            for chk in c.checks:
                mark = (
                    ":green[✓]"
                    if chk.value >= 0.999
                    else (":red[✗]" if chk.value == 0 else ":orange[~]")
                )
                st.caption(f"{mark} {chk.label}: {escape_markdown(chk.reason)}")
            with st.container(height=320, border=False):
                if c.error:
                    st.error(c.error)
                else:
                    st.markdown(demote_headings(escape_markdown(c.output or "")))

st.markdown("")
d1, d2, d3 = st.columns(3)
d1.download_button(
    "Download report (HTML)",
    html_report(bench, results, name),
    file_name=f"{bench.slug}-report.html",
    mime="text/html",
    key="dl-html",
    use_container_width=True,
)
csv = pd.DataFrame(
    [
        {
            "test_case": c.case_index + 1,
            "model": name(c.model_id),
            "score": c.score,
            "cost_usd": c.cost_usd,
            "latency_s": c.latency_s,
            "answer": c.output,
            "error": c.error,
        }
        for c in results.cases
    ]
).to_csv(index=False)
d2.download_button(
    "Download answers (CSV)",
    csv,
    file_name=f"{bench.slug}-answers.csv",
    mime="text/csv",
    key="dl-csv",
    use_container_width=True,
)
d3.download_button(
    "Download benchmark (YAML)",
    yaml.safe_dump(bench.to_dict(), sort_keys=False, allow_unicode=True),
    file_name=f"{bench.slug}.yaml",
    mime="text/yaml",
    key="dl-yaml",
    use_container_width=True,
)
if os.environ.get("ROUTEWISE_DEBUG"):
    st.json(bench.to_dict())
