"""A self-contained HTML report of a studio benchmark: share it, attach it, print it.

No scripts and no remote assets; every piece of text is escaped (answers are model output).
"""

from __future__ import annotations

import html
from collections.abc import Callable
from datetime import UTC, datetime

from routewise.studio.model import StudioBenchmark
from routewise.studio.run import StudioResults

_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
color:#0b0b0b;background:#fcfcfb;margin:0;line-height:1.5}
main{max-width:980px;margin:0 auto;padding:40px 24px 64px}
.eyebrow{color:#2a78d6;font-weight:700;font-size:12px;letter-spacing:.08em;text-transform:uppercase}
h1{font-size:30px;letter-spacing:-.02em;margin:6px 0 4px}
h2{font-size:19px;margin:34px 0 10px}
.sub{color:#52514e}
.verdict{border:1px solid #cde2fb;border-left:5px solid #2a78d6;border-radius:14px;
background:#f4f8fe;padding:18px 22px;margin:22px 0}
.verdict b{font-size:20px}
table{width:100%;border-collapse:collapse;font-size:14px;background:#fff;border:1px solid #e7e5e0;
border-radius:12px;overflow:hidden}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #eeede9;vertical-align:top}
th{background:#f3f2ef;color:#52514e;font-weight:600}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
tr.best td{background:#f4f8fe}
.case{background:#fff;border:1px solid #e7e5e0;border-radius:12px;padding:14px 16px;margin:12px 0}
.answer{white-space:pre-wrap;font-size:13px;background:#f8f8f6;border-radius:8px;padding:10px 12px;
max-height:none;overflow:visible}
.muted{color:#8a8984;font-size:12px}
.chip{display:inline-block;border-radius:999px;padding:2px 10px;font-size:12px;background:#f3f2ef;
border:1px solid #e7e5e0;margin-right:6px}
"""


def _e(x: object) -> str:
    return html.escape(str(x))


def html_report(bench: StudioBenchmark, results: StudioResults, name: Callable[[str], str]) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_e(bench.name)}: benchmark report</title><style>{_CSS}</style></head><body>",
        "<main>",
        "<div class='eyebrow'>routewise benchmark report</div>",
        f"<h1>{_e(bench.name)}</h1>",
        f"<div class='sub'>{_e(bench.description or '')} {results.total_cases} test cases, "
        f"{len(results.models)} models. Generated {when}.</div>",
    ]
    if results.models:
        best = results.models[0]
        value = next(m for m in results.models if m.model_id == results.best_value)
        if value.model_id == best.model_id:
            line = (
                f"{_e(name(best.model_id))} scored highest ({best.mean_score:.2f}) and is "
                "also the best value."
            )
        else:
            share = (value.mean_cost_usd / best.mean_cost_usd) if best.mean_cost_usd else 0
            line = (
                f"Best quality: {_e(name(best.model_id))} ({best.mean_score:.2f}). "
                f"Best value: {_e(name(value.model_id))} ({value.mean_score:.2f}) at "
                f"{share:.0%} of the cost."
            )
        parts.append(f"<div class='verdict'><b>{line}</b></div>")

        parts.append(
            "<h2>Results by model</h2><table><tr><th>Model</th><th>Score</th>"
            f"<th>Pass rate (≥{results.pass_threshold:.1f})</th><th>Cost / case</th>"
            "<th>Total cost</th><th>Latency</th></tr>"
        )
        for m in results.models:
            cls = " class='best'" if m.model_id == results.best_quality else ""
            parts.append(
                f"<tr{cls}><td>{_e(name(m.model_id))}</td><td class='num'>{m.mean_score:.2f}</td>"
                f"<td class='num'>{m.pass_rate:.0%}</td><td class='num'>${m.mean_cost_usd:.5f}</td>"
                f"<td class='num'>${m.total_cost_usd:.4f}</td>"
                f"<td class='num'>{m.mean_latency_s:.1f}s</td></tr>"
            )
        parts.append("</table>")

        labels = list(dict.fromkeys(k for m in results.models for k in m.checks))
        if labels:
            parts.append(
                "<h2>Results by check</h2><table><tr><th>Check</th>"
                + "".join(f"<th>{_e(name(m.model_id))}</th>" for m in results.models)
                + "</tr>"
            )
            for label in labels:
                cells = "".join(
                    f"<td class='num'>{m.checks[label]:.2f}</td>"
                    if label in m.checks
                    else "<td>-</td>"
                    for m in results.models
                )
                parts.append(f"<tr><td>{_e(label)}</td>{cells}</tr>")
            parts.append("</table>")

    parts.append("<h2>The task</h2>")
    if bench.system_prompt:
        parts.append(
            f"<div class='case'><div class='muted'>System prompt</div>"
            f"<div class='answer'>{_e(bench.system_prompt)}</div></div>"
        )
    parts.append(
        f"<div class='case'><div class='muted'>Prompt template</div>"
        f"<div class='answer'>{_e(bench.prompt)}</div></div>"
    )

    parts.append("<h2>Every answer</h2>")
    by_case: dict[int, list] = {}
    for c in results.cases:
        by_case.setdefault(c.case_index, []).append(c)
    cases = bench.effective_cases()
    for idx in sorted(by_case):
        values = cases[idx] if idx < len(cases) else {}
        chips = "".join(
            f"<span class='chip'>{_e(k)}: {_e(str(v)[:80])}</span>" for k, v in values.items()
        )
        parts.append(f"<div class='case'><b>Test case {idx + 1}</b><div>{chips}</div>")
        for c in by_case[idx]:
            score = "error" if c.score is None else f"{c.score:.2f}"
            reasons = "; ".join(f"{chk.label}: {chk.reason}" for chk in c.checks)
            body = c.output if c.output is not None else (c.error or "")
            parts.append(
                f"<p><b>{_e(name(c.model_id))}</b> <span class='chip'>score {score}</span>"
                f"<br><span class='muted'>{_e(reasons)}</span></p>"
                f"<div class='answer'>{_e(body)}</div>"
            )
        parts.append("</div>")
    parts.append(
        "<p class='muted'>Costs are list-price estimates. Scores are 0 to 1, averaged "
        "over the checks you chose.</p></main></body></html>"
    )
    return "".join(parts)
