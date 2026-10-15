"""Overview: what routewise is, the evaluate -> compare -> route loop, and where to start."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from routewise.bench.core import load_benchmarks
from routewise.bench.store import BenchStore
from routewise.config import default_db_path
from routewise.dashboard.ui import callout, hero, inject_css, kpis, render
from routewise.evals.loader import all_tasks
from routewise.evals.store import ResultStore
from routewise.models.registry import load_models
from routewise.settings import model_status
from routewise.studio.model import list_saved

HERE = Path(__file__).parent

inject_css()
render(
    """
<style>
.rw-loop { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 8px 0 6px; }
.rw-loop-card { background: #fff; border: 1px solid #e7e5e0; border-radius: 16px;
  padding: 20px 22px; }
.rw-loop-n { color: #2a78d6; font-weight: 700; font-size: .75rem; letter-spacing: .08em;
  text-transform: uppercase; }
.rw-loop-t { font-weight: 800; font-size: 1.25rem; color: #0b0b0b; margin: 4px 0 6px;
  letter-spacing: -.01em; }
.rw-loop-d { color: #52514e; font-size: .93rem; line-height: 1.5; }
.rw-loop-q { color: #184f95; font-size: .85rem; font-weight: 600; margin-top: 10px; }
@media (max-width: 760px) { .rw-loop { grid-template-columns: 1fr; } }
</style>
"""
)

render(
    hero(
        "routewise",
        "Know which AI model to use, for every task",
        "New models ship every few weeks and public leaderboards do not measure your work. "
        "routewise benchmarks models on your own prompts, lets your team judge answers blind, and "
        "routes each request to the cheapest model that meets your quality bar.",
    )
)

# --- live numbers ---------------------------------------------------------------------------------

registry = load_models(include_disabled=True)
ready = [m for m, spec in registry.items() if model_status(spec)[0] == "ready"]
try:
    rows = ResultStore(default_db_path()).rows()
except Exception:  # an unreadable database should not take the landing page down
    rows = []
try:
    sessions = BenchStore(default_db_path()).sessions()
except Exception:
    sessions = []
try:
    n_tasks = len(all_tasks(include_studio=True))
except Exception:
    n_tasks = 0
n_saved = len(list_saved())
n_blind = len(load_benchmarks())

render(
    kpis(
        [
            ("Models ready to test", str(len(ready)), f"of {len(registry)} in the catalogue"),
            ("Scored answers", f"{len(rows):,}", "stored and reused, never paid for twice"),
            ("Eval tasks", str(n_tasks), f"including {n_saved} from your studio benchmarks"),
            ("Blind comparisons", str(len(sessions)), f"{n_blind} blind tasks ready to run"),
        ]
    )
)

# --- the loop -------------------------------------------------------------------------------------

st.markdown("### How it works")
render(
    """
<div class="rw-loop">
  <div class="rw-loop-card"><div class="rw-loop-n">1 · Evaluate</div>
    <div class="rw-loop-t">Benchmark studio</div>
    <div class="rw-loop-d">Write your prompt once, add test cases, pick models and choose the
    checks that define a good answer: exact answers, JSON fields, word limits, banned phrases,
    an AI judge with your rubric, and more.</div>
    <div class="rw-loop-q">Answers: which model is best on my work, and which is best value?</div>
  </div>
  <div class="rw-loop-card"><div class="rw-loop-n">2 · Compare blind</div>
    <div class="rw-loop-t">Blind benchmarks</div>
    <div class="rw-loop-d">For taste and judgment calls, rate answers without knowing which
    model wrote them. Names are revealed only after every answer is scored, so brand loyalty
    cannot creep in.</div>
    <div class="rw-loop-q">Answers: is the new model actually better than the one I use?</div>
  </div>
  <div class="rw-loop-card"><div class="rw-loop-n">3 · Route</div>
    <div class="rw-loop-t">Model router</div>
    <div class="rw-loop-d">Every scored answer teaches the router. Each request goes to the
    cheapest model that meets your quality bar for that kind of task, with the evidence shown
    and a fallback ready.</div>
    <div class="rw-loop-q">Answers: what should this request cost, without losing quality?</div>
  </div>
</div>
"""
)

c1, c2, c3 = st.columns(3)
c1.page_link(str(HERE / "studio_page.py"), label="Build a benchmark", icon=":material/science:")
c2.page_link(
    str(HERE / "bench_page.py"), label="Run a blind comparison", icon=":material/visibility_off:"
)
c3.page_link(str(HERE / "router_page.py"), label="Try the router", icon=":material/alt_route:")

# --- why ------------------------------------------------------------------------------------------

st.markdown("### Why teams use it")
w1, w2, w3 = st.columns(3)
w1.markdown(
    "**Your work, not a leaderboard.** Public benchmarks saturate and rarely match your prompts. "
    "Bring your own tasks and your own definition of good."
)
w2.markdown(
    "**Decide in minutes when a model ships.** Saved benchmarks rerun on a new model with one "
    "click. Answers are cached, so you only pay for what is new."
)
w3.markdown(
    "**Spend less without guessing.** Most requests do not need the flagship model. The router "
    "proves which ones do, and a CI gate catches quality regressions."
)

st.markdown("### Start here")
if not ready:
    render(
        callout(
            "No model is ready yet. Open Models & keys to add an API key, or sign in to the "
            "claude CLI to use your Claude plan.",
            "accent",
        )
    )
    st.page_link(str(HERE / "keys_page.py"), label="Set up models", icon=":material/key:")
else:
    render(
        callout(
            "New here? Open the Benchmark studio, load the Word problems template, pick two "
            "models and press Run. It takes about a minute and costs a few cents.",
            "accent",
        )
    )
