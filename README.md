# routewise

An eval-gated LLM router. Every routing decision is backed by measured eval scores: routewise
runs a task suite against each model, records quality, latency and cost per task category, and
then sends each incoming query to the model with the best trade-off for that kind of query. It
ships with a live cost/latency/quality dashboard, an OpenAI-compatible endpoint, a regression
gate for CI, and tooling to grow the eval set without writing code.

![dashboard](docs/dashboard.png)

## What

- **Router.** `route(query, mode)` classifies the query (structured output, reasoning,
  summarization, voice, safety), looks up per-category eval results, and picks a model in
  `quality`, `balanced` or `cost` mode. It returns a plain-English rationale naming the
  category, the top two contenders, and why the winner won. A guard never picks a model whose
  measured quality is more than a set margin below the best.
- **Evals.** 110 tasks: 16 hand-written core tasks with gold answers and anchored judge rubrics,
  52 templated tasks in YAML packs, and 42 script-generated hard tasks whose gold answers are
  computed (code is executed, puzzles are brute-force verified).
- **Proof.** A leave-one-out replay of the eval set compares the router against always using
  the flagship, every fixed model, and an oracle. Bootstrap confidence intervals on every mean.
- **Operations.** OpenAI-compatible `/v1/chat/completions` with failover, a CI gate that fails
  on quality regressions, a budget cap, response caching, and an audit that finds broken gold
  answers.

## Quick start

```bash
uv sync
uv run routewise import            # load the shipped seed results (no API calls)
uv run routewise dashboard         # http://localhost:8501
uv run routewise route "Extract the order id and total as JSON" --mode balanced
uv run routewise report            # markdown tables
```

Run your own evals (see [Models and keys](#models-and-keys)):

```bash
uv run routewise eval --limit 4    # cheap smoke run: first 4 tasks x every enabled model
uv run routewise eval              # everything; reruns skip results already recorded
```

## Architecture

```
                         +----------------------------- eval loop -----------------------------+
                         |                                                                      |
  tasks (core + YAML     v                                                                      |
  packs + generated) --> runner --> chat() --> model --> scorers + LLM judge --> SQLite results -+
                                    (LiteLLM | claude CLI | mock)                     |
                                                                                      v
  query --> classifier ----------> leaderboard (per model x category: --> router --> model
            (neighbours over        mean quality, CI, cost, latency)     (mode weights,   (failover
             eval prompts,                                                quality guard)    to next)
             keyword rules,                                                   |
             optional LLM)                                                    +--> rationale
```

| Module | Role |
| --- | --- |
| `routewise/models/` | Model registry and prices (`config/models.yaml`), async `chat()` with retries, cache and three backends |
| `routewise/evals/` | Task format and scorers, built-in tasks, YAML pack loader, runner, result store, rescore, audit |
| `routewise/router/` | Classifier, leaderboard, routing policy (`config/router.yaml`), replay, report, gate |
| `routewise/dashboard/` | Streamlit app |
| `routewise/server.py` | OpenAI-compatible endpoint |
| `routewise/cli.py` | `routewise` command |

## Methodology

**Routing score.** For each candidate model in the query's category:

```
score = w_quality * quality - w_cost * cost / max_cost - w_latency * latency / max_latency
```

Weights per mode live in `config/router.yaml`. In `quality` mode cost and latency only break
near-ties. `balanced` and `cost` modes also refuse any model more than `max_quality_drop` below
the category's best (0.10 and 0.15). Categories with fewer than `min_samples` results per model
fall back to overall scores. Queries nothing recognizes are routed on overall scores too.

**Classifier.** The router classifies the whole request (system prompt plus user turn):

1. Nearest neighbours (TF-IDF cosine) over the labelled eval prompts, so every task you add
   also teaches the classifier.
2. Keyword rules, with matches in the first 200 characters (the instruction) counting double, so
   a pasted document full of the word "bug" does not turn a summary request into debugging.
3. Optionally, a cheap LLM for anything left over (`classifier.llm_fallback`).

**Scorers.** Every task lists its scorers as data, so the runner needs no per-task code.
Programmatic: `exact_match` (text, number, regex group, JSON fields), `schema_valid`,
`slot_f1` (dict slots or lists of items, with "contains" and "any of" gold values),
`number_preservation`, `word_count_cap`, `bullet_count`, `pii_absent`, `forbidden_strings`,
`contains_all`, `contains_any`, `regex_match`. Every score is a float in [0, 1] with a reason.
A component can be `required`: if it scores 0 the task scores 0, so a wrong answer cannot
earn half credit for nice prose.

**LLM judge.** For qualities code cannot check (tone, faithfulness, explanation quality), a
cheap judge model grades against an anchored 1-5 rubric stored on the task itself (so grading
is auditable in the diff), returns JSON, and is normalized to (score - 1) / 4. The response is
fenced and the judge is told to treat it as data, not instructions. An unparseable verdict is
recorded as an error and retried, never silently scored 0.

**Versioning.** A task's version hashes only what the model sees (prompt and system prompt).
Editing a prompt invalidates old results; fixing a gold answer or scorer does not, and
`routewise rescore` reapplies scorers to stored outputs at no model cost.

**Replay.** For each eval task the router decides with a leaderboard and neighbour index that
exclude that task (leave-one-out), then takes the recorded result of the model it chose.

**Audit.** `routewise audit` flags tasks where every model scores the same imperfect value
(independent models rarely make the identical mistake, so suspect the gold), tasks every model
aces (saturated), tasks where models disagree (the ones that drive routing), and tasks where
one model's repeated samples disagree (flaky). Its first run caught three gold-answer bugs in
this repo's own task packs.

## Results

Seeded on 2026-10-14 against Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5: 110 tasks, 504 results
(the core and hard suites were sampled twice). Full tables in [results/report.md](results/report.md).

| Model | Mean quality (95% CI) | Mean cost / task | Total cost | n |
|---|---|---|---|---|
| claude-opus-5-5 | 1.00 (1.00-1.00) | $0.00613 | $1.0290 | 168 |
| claude-sonnet-5-5 | 1.00 (0.99-1.00) | $0.00367 | $0.6161 | 168 |
| claude-haiku-5-5 | 0.99 (0.99-1.00) | $0.00035 | $0.0584 | 168 |

| Model | reasoning | safety | structured_output | summarization | voice | overall |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 | 1.00 | 1.00 | 1.00 | 0.95 | 0.98 | 0.99 |
| claude-opus-5-5 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 |
| claude-sonnet-5-5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 |

Router replay (leave-one-out) against always using the flagship (Opus):

| Mode | Router quality | Flagship quality | Router cost | Flagship cost | Savings |
|---|---|---|---|---|---|
| quality | 0.997 | 0.999 | $0.0688 | $0.5605 | 88% |
| balanced | 0.995 | 0.999 | $0.0331 | $0.5605 | 94% |
| cost | 0.995 | 0.999 | $0.0331 | $0.5605 | 94% |

Classifier accuracy on the 110 tasks (leave-one-out): 97%.

**Interpretation.** On this task set the three Claude 5.5 models are nearly indistinguishable:
104 of 110 tasks are saturated (every model perfect), including all 42 generated hard tasks
(code tracing, calendar arithmetic, five- and seven-person logic puzzles, 4-digit
multiplication chains, constrained writing). So the router saves money by sending almost
everything to Haiku, about 94% cheaper than always using Opus for a quality loss of 0.003,
which is inside the noise. The flagship still wins where judgement and tight constraints meet:
summarization (Haiku 0.95 vs Opus 0.99 and Sonnet 1.00, mostly one Haiku TL;DR that ran 61
words against a 60-word cap) and voice (Opus 1.00 vs Haiku 0.98 and Sonnet 0.97, mostly the
judged tone of the angry-customer reply and the barge-in response). But the audit shows every one of those six discriminative tasks
is also flaky across samples, so these gaps are not yet statistically meaningful. The honest
conclusion: for tasks like these, the router's job is less "pick the smart model" and more
"prove the cheap model is safe", and the eval set needs harder, more varied tasks before it can
justify paying for Opus on any category.

### How the seed results were produced

No API keys were used. Claude models ran through the local `claude -p` CLI on a Claude plan
(the `claude_code` backend, tools disabled, an empty working directory, a minimal system
prompt). Consequences:

- **Costs are list-price equivalents**, computed from `config/models.yaml`, not money spent.
- **Input tokens are estimated** (about 4 characters per token) because the CLI adds its own
  scaffolding tokens. Output tokens come from the CLI and include extended-thinking tokens.
- **Latency is the API time** the CLI reports, including thinking time, so it is higher and
  flatter across models than direct API calls would be. Do not read the latency columns as
  API benchmarks.
- Rows carry `backend` and `tokens_estimated` so API results can be told apart later.

## Models and keys

`config/models.yaml` lists each model with prices per 1M tokens. With `backend: auto`, a model
uses its provider API key through LiteLLM when one is set (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, see `.env.example`), and otherwise falls back to the
`claude` CLI for Claude models. `ROUTEWISE_BACKEND` forces one backend for everything.

Claude prices are verified against the list-price cost basis the CLI reports. The OpenAI and
Gemini entries (model ids and prices) come from third-party trackers, are marked
`price_verified: false`, and are disabled until you add keys and check them against the
providers' pricing pages. `routewise models` shows each model's backend and price status.

## Scaling the evals

- **Task packs.** Drop a YAML file in `data/tasks/`. A pack holds static `tasks` and
  `templates` that expand over inline `cases` or a JSONL `cases_file`, with `{{placeholder}}`
  substitution (a value that is exactly `{{name}}` keeps its type).
- **Generators.** `scripts/generate_hard_pack.py` shows how to produce tasks whose gold answers
  are computed rather than hand-written; tests re-verify them.
- **Runs.** `--suite`, `--category`, `--tag`, `--task`, `--limit`, `--models`, `--repeats N`
  for variance, `--concurrency`, `--max-cost` budget cap, `--dry-run`. Runs stop cleanly when a
  plan or credit limit is hit; errored rows retry on the next run.
- **Quality control.** `routewise audit` after every run; `routewise rescore` after fixing
  gold; `routewise export` / `import` to share results without the database.

## Serving

```bash
uv run routewise serve --port 8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="routewise/balanced",  # or routewise/quality, routewise/cost, or a concrete model id
    messages=[{"role": "user", "content": "Summarize this thread: ..."}],
)
```

Responses carry a `routewise` object (mode, category, rationale, failover attempts, cost) and
`x-routewise-model` / `x-routewise-category` headers. If the chosen model errors, the request
fails over down the ranking. Streaming is not supported yet.

## CI and the eval gate

`.github/workflows/ci.yml` runs lint, format and the offline test suite (no API calls) on every
push and pull request. A manual `workflow_dispatch` job runs the core evals against real APIs
with a budget cap and fails if any (model, category) quality fell more than 0.05 below
`results/baseline-core.json`:

```bash
uv run routewise snapshot --suite core --out results/baseline-core.json   # accept a new baseline
uv run routewise gate --suite core --baseline results/baseline-core.json  # check against it
```

## Adding a model

1. Add an entry to `config/models.yaml`: `id`, `provider`, `litellm_model`, prices per 1M
   input and output tokens, and `cli_model` if it can run through the claude CLI.
2. Set the provider key in `.env`.
3. `uv run routewise eval --models <id> --limit 4`, then the full run. Existing results for
   other models are reused.
4. `uv run routewise report` and `routewise audit`. The router picks the new model up
   automatically once it has at least `min_samples` results per category.

## Adding an eval

1. Pick a category: `structured_output`, `reasoning`, `summarization`, `voice`, `safety`.
2. Add a task to a pack in `data/tasks/` (or a new pack). Give it a prompt, scorers, and a
   `reference_output` (a known-good answer). The test suite checks every reference scores 1.0,
   which catches broken gold data before any money is spent.
3. Prefer programmatic scorers. Use `llm_judge` with an anchored rubric (anchors for 5, 3 and
   1 at least) only for qualities code cannot check.
4. `uv run pytest`, then `uv run routewise eval --task <id>`.

## Development

```bash
uv sync
uv run pytest          # offline; mocks every model call
uv run ruff check . && uv run ruff format --check .
```

## Limitations

- **Small, mostly saturated eval set.** 110 tasks, most of them easy for current Claude models.
  Results are directional; per-category means rest on 10 (safety) to 92 (reasoning) results
  per model.
- **Judge noise and bias.** A single judge (Haiku 5.5 by default) grades its own family,
  including itself. Swap `--judge-model`, or average several judges, before trusting
  judge-heavy categories.
- **Seeded via the CLI**, so latency and input-token figures are approximations (see above).
- **Templated siblings** share wording, which flatters leave-one-out classifier accuracy.
- **Category-level routing.** The router uses per-category means, not per-query difficulty.
- **Only Claude models seeded.** OpenAI and Gemini entries are ready but unverified and
  disabled until keys are added.
