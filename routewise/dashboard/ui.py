"""Design system for the dashboard: tokens, one stylesheet, and small HTML components.

Colours come from the dataviz reference palette: one accent (blue, slot 1) used for emphasis
(the recommended or preferred model), recessive grays for everything else, and the single-hue
blue ramp for magnitudes. Every helper escapes its text, because some of it is model output.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence

import streamlit as st

ACCENT = "#2a78d6"
ACCENT_SOFT = "#cde2fb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
SURFACE = "#fcfcfb"
CARD = "#ffffff"
LINE = "#e7e5e0"
MUTED = "#f3f2ef"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
GRAY_MARK = "#c9c7c0"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, input, textarea, h1, h2, h3, h4, h5, h6,
button p, [data-testid="stWidgetLabel"] {{
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
h1, h2, h3 {{ letter-spacing: -0.02em; color: {INK}; }}
[data-testid="stSidebar"] {{ background: {MUTED}; border-right: 1px solid {LINE}; }}

.rw-hero {{ padding: 4px 0 18px; }}
.rw-eyebrow {{ color: {ACCENT}; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 6px; }}
.rw-title {{ font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em; color: {INK};
  line-height: 1.1; margin: 0; }}
.rw-sub {{ font-size: 1.08rem; color: {INK_2}; max-width: 760px; margin-top: 10px;
  line-height: 1.55; }}

.rw-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px; margin: 6px 0 26px; }}
.rw-kpi {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px;
  padding: 16px 18px; }}
.rw-kpi-label {{ color: {INK_2}; font-size: 0.8rem; font-weight: 500; }}
.rw-kpi-value {{ color: {INK}; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em;
  margin-top: 2px; }}
.rw-kpi-note {{ color: {INK_3}; font-size: 0.78rem; margin-top: 2px; }}

.rw-step {{ display: flex; align-items: flex-start; gap: 14px; margin: 30px 0 12px; }}
.rw-step-num {{ flex: 0 0 32px; height: 32px; border-radius: 50%; background: {ACCENT};
  color: white; font-weight: 700; display: flex; align-items: center; justify-content: center;
  font-size: 0.95rem; margin-top: 2px; }}
.rw-step-title {{ font-size: 1.3rem; font-weight: 700; color: {INK}; letter-spacing: -0.01em; }}
.rw-step-sub {{ color: {INK_2}; font-size: 0.95rem; margin-top: 2px; }}

.rw-verdict {{ background: linear-gradient(135deg, #f4f8fe 0%, {CARD} 70%);
  border: 1px solid {ACCENT_SOFT}; border-left: 5px solid {ACCENT}; border-radius: 16px;
  padding: 22px 26px; margin: 6px 0 8px; }}
.rw-verdict-head {{ font-size: 1.6rem; font-weight: 800; color: {INK};
  letter-spacing: -0.02em; }}
.rw-verdict-reason {{ font-size: 1.05rem; color: {INK_2}; margin-top: 6px; line-height: 1.5; }}
.rw-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
.rw-chip {{ background: {MUTED}; border: 1px solid {LINE}; color: {INK_2}; border-radius: 999px;
  padding: 4px 12px; font-size: 0.82rem; font-weight: 500; }}
.rw-chip.accent {{ background: #eef5fd; border-color: {ACCENT_SOFT}; color: #184f95; }}
.rw-chip.good {{ background: #edf8ed; border-color: #bfe5bf; color: #0b6b0b; }}

.rw-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px;
  padding: 16px 18px; height: 100%; }}
.rw-card.chosen {{ border: 2px solid {ACCENT}; box-shadow: 0 0 0 4px rgba(42,120,214,0.08); }}
.rw-card.dim {{ opacity: 0.62; }}
.rw-card-top {{ display: flex; justify-content: space-between; align-items: flex-start;
  gap: 8px; }}
.rw-card-name {{ font-weight: 700; color: {INK}; font-size: 1.02rem; }}
.rw-card-provider {{ color: {INK_3}; font-size: 0.8rem; }}
.rw-badge {{ font-size: 0.72rem; font-weight: 700; border-radius: 6px; padding: 3px 8px;
  white-space: nowrap; background: {MUTED}; color: {INK_2}; }}
.rw-badge.accent {{ background: {ACCENT}; color: white; }}
.rw-badge.warn {{ background: #fdf0ef; color: #a32b2b; }}
.rw-stat {{ margin-top: 12px; }}
.rw-stat-row {{ display: flex; justify-content: space-between; font-size: 0.85rem; }}
.rw-stat-label {{ color: {INK_2}; }}
.rw-stat-value {{ color: {INK}; font-weight: 600; font-variant-numeric: tabular-nums; }}
.rw-meter {{ height: 6px; background: {MUTED}; border-radius: 999px; margin-top: 5px;
  overflow: hidden; }}
.rw-meter > span {{ display: block; height: 100%; border-radius: 999px; background: {GRAY_MARK}; }}
.rw-card.chosen .rw-meter > span {{ background: {ACCENT}; }}
.rw-foot {{ color: {INK_3}; font-size: 0.78rem; margin-top: 12px; }}

.rw-callout {{ background: {MUTED}; border: 1px solid {LINE}; border-radius: 12px;
  padding: 12px 16px; color: {INK_2}; font-size: 0.92rem; line-height: 1.5; }}
.rw-callout.accent {{ background: #f4f8fe; border-color: {ACCENT_SOFT}; color: #1c3e66; }}

.rw-process {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
  margin: 4px 0 22px; }}
.rw-process-step {{ border: 1px solid {LINE}; background: {CARD}; border-radius: 12px;
  padding: 12px 14px; }}
.rw-process-step.active {{ border-color: {ACCENT}; background: #f4f8fe; }}
.rw-process-step.done {{ background: {MUTED}; }}
.rw-process-n {{ font-size: 0.75rem; font-weight: 700; color: {ACCENT}; letter-spacing: 0.06em;
  text-transform: uppercase; }}
.rw-process-t {{ font-weight: 700; color: {INK}; margin-top: 2px; }}
.rw-process-d {{ color: {INK_2}; font-size: 0.84rem; margin-top: 2px; }}

.rw-resp-label {{ font-weight: 800; font-size: 1.05rem; color: {INK}; }}
div[data-testid="stForm"] {{ border: none; padding: 0; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _e(text: object) -> str:
    return html.escape(str(text))


def render(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str) -> str:
    return (
        f'<div class="rw-hero"><div class="rw-eyebrow">{_e(eyebrow)}</div>'
        f'<h1 class="rw-title">{_e(title)}</h1><div class="rw-sub">{_e(subtitle)}</div></div>'
    )


def kpis(items: Iterable[tuple[str, str, str]]) -> str:
    tiles = "".join(
        f'<div class="rw-kpi"><div class="rw-kpi-label">{_e(label)}</div>'
        f'<div class="rw-kpi-value">{_e(value)}</div><div class="rw-kpi-note">{_e(note)}</div>'
        f"</div>"
        for label, value, note in items
    )
    return f'<div class="rw-kpis">{tiles}</div>'


def step(n: int | str, title: str, subtitle: str = "") -> str:
    sub = f'<div class="rw-step-sub">{_e(subtitle)}</div>' if subtitle else ""
    return (
        f'<div class="rw-step"><div class="rw-step-num">{_e(n)}</div>'
        f'<div><div class="rw-step-title">{_e(title)}</div>{sub}</div></div>'
    )


def chips(items: Iterable[tuple[str, str]]) -> str:
    """items: (text, kind) where kind is '', 'accent' or 'good'."""
    inner = "".join(f'<span class="rw-chip {_e(kind)}">{_e(text)}</span>' for text, kind in items)
    return f'<div class="rw-chips">{inner}</div>'


def verdict_card(headline: str, reason: str, chip_items: Sequence[tuple[str, str]] = ()) -> str:
    return (
        f'<div class="rw-verdict"><div class="rw-verdict-head">{_e(headline)}</div>'
        f'<div class="rw-verdict-reason">{_e(reason)}</div>'
        f"{chips(chip_items) if chip_items else ''}</div>"
    )


def model_card(
    *,
    name: str,
    provider: str,
    badge: str,
    stats: Sequence[tuple[str, str, float | None]],
    footnote: str = "",
    kind: str = "",
    badge_kind: str = "",
) -> str:
    """A model's expected performance. stats: (label, value, meter fraction 0-1 or None).
    kind: '' | 'chosen' | 'dim'."""
    rows = []
    for label, value, frac in stats:
        meter = (
            f'<div class="rw-meter"><span style="width:{max(0.0, min(1.0, frac)) * 100:.1f}%">'
            f"</span></div>"
            if frac is not None
            else ""
        )
        rows.append(
            f'<div class="rw-stat"><div class="rw-stat-row"><span class="rw-stat-label">'
            f'{_e(label)}</span><span class="rw-stat-value">{_e(value)}</span></div>{meter}</div>'
        )
    foot = f'<div class="rw-foot">{_e(footnote)}</div>' if footnote else ""
    return (
        f'<div class="rw-card {_e(kind)}"><div class="rw-card-top"><div>'
        f'<div class="rw-card-name">{_e(name)}</div>'
        f'<div class="rw-card-provider">{_e(provider)}</div></div>'
        f'<span class="rw-badge {_e(badge_kind)}">{_e(badge)}</span></div>'
        f"{''.join(rows)}{foot}</div>"
    )


def callout(text: str, kind: str = "") -> str:
    return f'<div class="rw-callout {_e(kind)}">{_e(text)}</div>'


def process(steps: Sequence[tuple[str, str]], active: int, done: Iterable[int] = ()) -> str:
    done = set(done)
    cells = []
    for i, (title, desc) in enumerate(steps):
        state = "active" if i == active else ("done" if i in done else "")
        cells.append(
            f'<div class="rw-process-step {state}"><div class="rw-process-n">Step {i + 1}'
            f'{" ✓" if i in done else ""}</div><div class="rw-process-t">{_e(title)}</div>'
            f'<div class="rw-process-d">{_e(desc)}</div></div>'
        )
    return f'<div class="rw-process">{"".join(cells)}</div>'


def money(x: float) -> str:
    if x == 0:
        return "$0"
    if x < 0.01:
        return f"${x:.5f}"
    return f"${x:,.2f}"
