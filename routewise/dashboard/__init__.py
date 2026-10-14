"""Streamlit dashboard."""


def escape_markdown(text: str) -> str:
    """Escape characters Streamlit markdown would misread (a pair of $ becomes LaTeX)."""
    return text.replace("$", "\\$")
