"""routewise dashboard.

    uv run routewise dashboard      (or: uv run streamlit run routewise/dashboard/app.py)

Pages: Overview, Benchmark studio (your prompts, your checks, any models), Blind benchmarks
(human-rated, model-blind comparisons), Model router, and Models & keys.
Reads results from data/results.db (or $ROUTEWISE_DB).
"""

from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="routewise", page_icon=":material/alt_route:", layout="wide")
page = st.navigation(
    {
        "": [
            st.Page(
                str(HERE / "overview_page.py"),
                title="Overview",
                icon=":material/home:",
                default=True,
            ),
        ],
        "Evaluate": [
            st.Page(
                str(HERE / "studio_page.py"), title="Benchmark studio", icon=":material/science:"
            ),
            st.Page(
                str(HERE / "bench_page.py"),
                title="Blind benchmarks",
                icon=":material/visibility_off:",
            ),
        ],
        "Use": [
            st.Page(
                str(HERE / "router_page.py"), title="Model router", icon=":material/alt_route:"
            ),
        ],
        "Settings": [
            st.Page(str(HERE / "keys_page.py"), title="Models & keys", icon=":material/key:"),
        ],
    }
)
page.run()
