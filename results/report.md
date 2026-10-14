### Per-model results

| Model | Mean quality (95% CI) | Mean latency | Mean cost / task | Total cost | n |
|---|---|---|---|---|---|
| claude-opus-5-5 | 1.00 (1.00-1.00) | 3.5s | $0.00613 | $1.0290 | 168 |
| claude-sonnet-5-5 | 1.00 (0.99-1.00) | 3.5s | $0.00367 | $0.6161 | 168 |
| claude-haiku-5-5 | 0.99 (0.99-1.00) | 3.5s | $0.00035 | $0.0584 | 168 |

### Mean quality by category

| Model | reasoning | safety | structured_output | summarization | voice | overall |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 | 1.00 | 1.00 | 1.00 | 0.95 | 0.98 | 0.99 |
| claude-opus-5-5 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 |
| claude-sonnet-5-5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 |

### Router replay (leave-one-out) vs always-flagship

| Mode | Policy | Mean quality | Total cost | Cost vs flagship | Picks |
|---|---|---|---|---|---|
| quality | router | 0.997 | $0.0688 | -88% | claude-haiku-5-5 91, claude-opus-5-5 12, claude-sonnet-5-5 7 |
| quality | flagship | 0.999 | $0.5605 | +0% | claude-opus-5-5 110 |
| quality | oracle | 1.000 | $0.0432 | -92% | claude-haiku-5-5 106, claude-sonnet-5-5 3, claude-opus-5-5 1 |
| balanced | router | 0.995 | $0.0331 | -94% | claude-haiku-5-5 110 |
| balanced | flagship | 0.999 | $0.5605 | +0% | claude-opus-5-5 110 |
| balanced | oracle | 1.000 | $0.0432 | -92% | claude-haiku-5-5 106, claude-sonnet-5-5 3, claude-opus-5-5 1 |
| cost | router | 0.995 | $0.0331 | -94% | claude-haiku-5-5 110 |
| cost | flagship | 0.999 | $0.5605 | +0% | claude-opus-5-5 110 |
| cost | oracle | 1.000 | $0.0432 | -92% | claude-haiku-5-5 106, claude-sonnet-5-5 3, claude-opus-5-5 1 |

- **quality**: 110 tasks, cost savings vs flagship 88%, quality delta -0.002, classifier accuracy 97%

- **balanced**: 110 tasks, cost savings vs flagship 94%, quality delta -0.003, classifier accuracy 97%

- **cost**: 110 tasks, cost savings vs flagship 94%, quality delta -0.003, classifier accuracy 97%
