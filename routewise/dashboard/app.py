"""routewise dashboard.

    uv run routewise dashboard      (or: uv run streamlit run routewise/dashboard/app.py)

Pages: the eval-gated Router, and Blind benchmarks (human-rated, model-blind comparisons).
Reads results from data/results.db (or $ROUTEWISE_DB).
"""

from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="routewise", layout="wide")
page = st.navigation(
    [
        st.Page(str(HERE / "router_page.py"), title="Router", default=True),
        st.Page(str(HERE / "bench_page.py"), title="Blind benchmarks"),
    ]
)
page.run()
