"""Streamlit dashboard."""


def escape_markdown(text: str) -> str:
    """Escape characters Streamlit markdown would misread (a pair of $ becomes LaTeX)."""
    return text.replace("$", "\\$")


def demote_headings(text: str, level: int = 5) -> str:
    """Render a response's markdown headings at a small fixed level (outside code fences) so a
    model's '# Title' does not dwarf the page around it."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            stripped = line.lstrip("#")
            hashes = len(line) - len(stripped)
            if 1 <= hashes <= 6 and stripped.startswith(" "):
                line = "#" * level + stripped
        out.append(line)
    return "\n".join(out)
