"""Starter benchmarks: realistic, ready to run, and easy to adapt."""

from __future__ import annotations

from routewise.studio.catalog import DEFAULT_RUBRIC
from routewise.studio.model import CheckConfig, StudioBenchmark

SUPPORT_RUBRIC = "\n".join(
    [
        "5: Acknowledges the specific problem, sincere without grovelling, gives a concrete next "
        "step, promises nothing it cannot guarantee, under the word limit.",
        "4: Empathetic and correct, but the next step is vague or the tone slightly stiff.",
        "3: Polite but generic; could be sent to any customer.",
        "2: Defensive, dismissive, or makes an implied promise.",
        "1: Rude, wrong, or makes explicit promises about refunds or dates.",
    ]
)

FAITHFUL_RUBRIC = """5: Every claim is supported by the source; nothing important is missing.
4: Faithful, but omits one secondary point.
3: Faithful but vague, or misses an important point.
2: Contains one unsupported or wrong claim.
1: Several wrong or invented claims."""

TEMPLATES: list[StudioBenchmark] = [
    StudioBenchmark(
        name="Invoice extraction",
        description="Pull structured fields out of messy invoice emails.",
        category="structured_output",
        prompt=(
            "Extract the invoice number, issue date (YYYY-MM-DD) and total (a number, no "
            "currency symbol) from this email. Reply with only JSON using the keys "
            '"invoice_number", "date" and "total".\n\n{{email}}'
        ),
        cases=[
            {
                "email": "Hi team, invoice INV-2291 dated 3 March 2026 comes to $1,240.00. Thanks, "
                "Priya",
                "expected": '{"invoice_number": "INV-2291", "date": "2026-03-03", "total": 1240}',
            },
            {
                "email": "Attached: #A-77 (issued 2026-02-14). Amount due 89.50 USD within 30 "
                "days.",
                "expected": '{"invoice_number": "A-77", "date": "2026-02-14", "total": 89.5}',
            },
            {
                "email": "Reminder about bill 5521 from Jan 9th 2026, total of EUR 3,000 still "
                "open. A separate credit note of 200 was issued.",
                "expected": '{"invoice_number": "5521", "date": "2026-01-09", "total": 3000}',
            },
        ],
        checks=[
            CheckConfig(
                "json_schema",
                {
                    "schema": (
                        '{"type": "object", "required": ["invoice_number", "date", "total"], '
                        '"properties": {"total": {"type": "number"}}}'
                    )
                },
            ),
            CheckConfig("json_fields", {}, weight=2.0),
        ],
    ),
    StudioBenchmark(
        name="Support replies",
        description="Short, empathetic replies that never over-promise.",
        category="voice",
        system_prompt=(
            "You are a support agent for {{company}}. You cannot authorise refunds, "
            "replacements or delivery dates. Reply in under 60 words."
        ),
        prompt="Customer message:\n{{message}}",
        cases=[
            {
                "company": "Northwind Kitchen",
                "message": "Third time my blender arrived "
                "cracked. I want this fixed TODAY or I'm done with you.",
            },
            {
                "company": "Lumen Bank",
                "message": "Why was I charged a $35 fee? Nobody told me "
                "about this and I'm furious.",
            },
            {
                "company": "Tinkerly",
                "message": "Your app deleted my project an hour before my "
                "client demo. What are you going to do about it?",
            },
        ],
        checks=[
            CheckConfig("word_limit", {"max_words": 60, "min_words": 10}),
            CheckConfig(
                "forbidden",
                {"words": "guarantee, I promise, full refund, by tomorrow, within 24 hours"},
                required=True,
            ),
            CheckConfig("llm_judge", {"rubric": SUPPORT_RUBRIC}, weight=2.0),
        ],
    ),
    StudioBenchmark(
        name="Faithful summaries",
        description="Summaries that keep every number and invent nothing.",
        category="summarization",
        prompt="Summarize this in at most 2 sentences. Keep every number exactly.\n\n{{text}}",
        cases=[
            {
                "text": "Q3 revenue was $4.8M, up 22% year over year, while churn fell from 3.1% "
                "to 2.4%. We added 140 net new customers and expect Q4 revenue between "
                "$5.1M and $5.4M."
            },
            {
                "text": "The pilot enrolled 312 patients across 9 sites. Median wait time dropped "
                "from 47 to 18 minutes and no-shows fell by 35%, but staff overtime rose 6% "
                "in the first 8 weeks."
            },
        ],
        checks=[
            CheckConfig("keeps_numbers", {"column": "text"}, weight=2.0),
            CheckConfig("word_limit", {"max_words": 60}),
            CheckConfig("llm_judge", {"rubric": FAITHFUL_RUBRIC}),
        ],
    ),
    StudioBenchmark(
        name="Word problems",
        description="Multi-step arithmetic with a single checkable answer.",
        category="reasoning",
        prompt="{{question}}\nShow brief working, then give the final line as 'ANSWER: <number>'.",
        cases=[
            {
                "question": "A warehouse holds 1,200 units, ships 15% on Monday, receives 345 on "
                "Tuesday, then ships half of what it holds (rounded down) on "
                "Wednesday. How many units remain?",
                "expected": "683",
            },
            {
                "question": "18 licenses cost $45 each with a 25% discount, plus a flat $150 setup "
                "fee. What is the total in dollars?",
                "expected": "757.5",
            },
            {
                "question": "A train leaves at 09:40 and arrives at 13:05. A second train takes 35 "
                "minutes longer. How many minutes does the second train take?",
                "expected": "240",
            },
        ],
        checks=[
            CheckConfig("expected_answer", {"mode": "number"}, required=True),
            CheckConfig("regex", {"pattern": r"ANSWER:\s*[\d.,]+"}),
        ],
    ),
    StudioBenchmark(
        name="Blank benchmark",
        description="Start from scratch.",
        category="reasoning",
        prompt="{{input}}",
        cases=[{"input": "", "expected": ""}],
        checks=[CheckConfig("llm_judge", {"rubric": DEFAULT_RUBRIC})],
    ),
]

STARTERS = [t for t in TEMPLATES if t.name != "Blank benchmark"]
