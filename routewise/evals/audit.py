"""Eval QA: find tasks that are probably broken, too easy, or noisy.

- suspect gold: every model scores the same imperfect value. Independent models rarely make
  the identical mistake, so the gold answer or scorer is the likelier culprit.
- saturated: every model is perfect. Fine as a regression check, useless for routing.
- discriminative: models disagree. These are the tasks that drive routing decisions.
- flaky: one model's repeated samples disagree. Its mean is noisy; add samples.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from routewise.evals.store import ResultRow


@dataclass
class TaskFinding:
    task_id: str
    model_means: dict[str, float]

    @property
    def spread(self) -> float:
        return max(self.model_means.values()) - min(self.model_means.values())


@dataclass
class FlakyFinding:
    task_id: str
    model_id: str
    samples: list[float]


@dataclass
class AuditReport:
    suspect_gold: list[TaskFinding] = field(default_factory=list)
    saturated: list[str] = field(default_factory=list)
    discriminative: list[TaskFinding] = field(default_factory=list)
    flaky: list[FlakyFinding] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"suspect gold ({len(self.suspect_gold)}): every model equally imperfect, check the "
            "gold answer or scorer"
        ]
        for f in self.suspect_gold:
            lines.append(f"  {f.task_id}: all models {next(iter(f.model_means.values())):.2f}")
        lines.append(f"discriminative ({len(self.discriminative)}): models disagree")
        for f in sorted(self.discriminative, key=lambda f: -f.spread):
            means = ", ".join(f"{m} {q:.2f}" for m, q in sorted(f.model_means.items()))
            lines.append(f"  {f.task_id}: spread {f.spread:.2f} ({means})")
        lines.append(f"flaky ({len(self.flaky)}): a model's samples disagree")
        for f in self.flaky:
            lines.append(f"  {f.task_id} x {f.model_id}: {[round(x, 2) for x in f.samples]}")
        lines.append(f"saturated ({len(self.saturated)}): every model perfect")
        return "\n".join(lines)


def audit(rows: Iterable[ResultRow], *, tolerance: float = 0.01) -> AuditReport:
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.quality is not None and not r.error:
            samples[r.task_id][r.model_id].append(r.quality)
    report = AuditReport()
    for task_id, by_model in samples.items():
        for model_id, values in by_model.items():
            if len(values) > 1 and max(values) - min(values) > tolerance:
                report.flaky.append(FlakyFinding(task_id, model_id, values))
        if len(by_model) < 2:
            continue
        finding = TaskFinding(task_id, {m: sum(v) / len(v) for m, v in by_model.items()})
        if finding.spread > tolerance:
            report.discriminative.append(finding)
        elif min(finding.model_means.values()) >= 1 - tolerance:
            report.saturated.append(task_id)
        else:
            report.suspect_gold.append(finding)
    return report
