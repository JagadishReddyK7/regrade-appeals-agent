"""Statistics for comparing runs (proposal section 11, protocol).

- Paired permutation test: two runs on the same cases; under the null hypothesis the two
  systems are exchangeable, so each unit's records are swapped between the runs at random
  and the metric difference recomputed. Units are grading situations by default: the tone
  variants of one situation are correlated, so they are swapped together.
- Holm correction across the comparisons of one family (the pre-registered comparisons).
- Bootstrap 95% confidence intervals, resampling grading situations with replacement.

Every function takes any metric from eval.metrics. Resamples whose metric is None (empty
denominator) are skipped and counted. Pure Python with a seeded random.Random.

Usage: python -m eval.stats <run dir A> <run dir B> [<run dir B2> ...]
       [--metrics wrongful_change_rate wrongful_refusal_rate] [--resamples 10000] [--seed 0]
       [--allow-test-split]
compares A with every B on every metric, with Holm correction across all comparisons.
"""

import argparse
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from agent.schemas import Decision
from eval.metrics import METRICS, CaseRecord, Metric
from eval.run import load_cases

Unit = Callable[[CaseRecord], str]
PRE_REGISTERED = ("wrongful_change_rate", "wrongful_refusal_rate")


def by_situation(record: CaseRecord) -> str:
    return record.situation_id


def group_units(records: Sequence[CaseRecord], unit: Unit) -> dict[str, list[CaseRecord]]:
    groups: dict[str, list[CaseRecord]] = defaultdict(list)
    for r in records:
        groups[unit(r)].append(r)
    return groups


def difference(metric: Metric, a: Sequence[CaseRecord], b: Sequence[CaseRecord]) -> float | None:
    value_a, value_b = metric(a), metric(b)
    return None if value_a is None or value_b is None else value_a - value_b


# --- Paired permutation test ------------------------------------------------------------


@dataclass(frozen=True)
class PermutationResult:
    difference: float | None  # metric(a) - metric(b); None if either is undefined
    p_value: float | None  # two-sided
    n_resamples: int  # resamples with a defined difference
    exact: bool  # every swap pattern enumerated instead of sampled


def paired_units(
    a: Sequence[CaseRecord], b: Sequence[CaseRecord], unit: Unit
) -> list[tuple[list[CaseRecord], list[CaseRecord]]]:
    """Each unit's records from both runs, in the same case order."""
    ids_a, ids_b = [r.case_id for r in a], [r.case_id for r in b]
    if len(set(ids_a)) != len(ids_a) or set(ids_a) != set(ids_b) or len(ids_a) != len(ids_b):
        raise ValueError("paired tests need two runs on exactly the same cases")
    b_by_id = {r.case_id: r for r in b}
    pairs = []
    for _, records in sorted(group_units(a, unit).items()):
        pairs.append((records, [b_by_id[r.case_id] for r in records]))
    return pairs


def swapped(
    pairs: Sequence[tuple[list[CaseRecord], list[CaseRecord]]], flips: Sequence[bool]
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    a: list[CaseRecord] = []
    b: list[CaseRecord] = []
    for (unit_a, unit_b), flip in zip(pairs, flips, strict=True):
        a.extend(unit_b if flip else unit_a)
        b.extend(unit_a if flip else unit_b)
    return a, b


def paired_permutation_test(
    a: Sequence[CaseRecord],
    b: Sequence[CaseRecord],
    metric: Metric,
    n_resamples: int = 10_000,
    seed: int = 0,
    unit: Unit = by_situation,
) -> PermutationResult:
    """Two-sided p-value for metric(a) != metric(b). With at most n_resamples swap patterns
    all are enumerated (exact test); otherwise random swaps with p = (1 + k) / (1 + n)."""
    pairs = paired_units(a, b, unit)
    observed = difference(metric, a, b)
    exact = 2 ** len(pairs) <= n_resamples
    if observed is None:
        return PermutationResult(None, None, 0, exact)
    if exact:
        patterns: Sequence[Sequence[bool]] = list(
            itertools.product([False, True], repeat=len(pairs))
        )
    else:
        rng = random.Random(seed)
        patterns = [[rng.random() < 0.5 for _ in pairs] for _ in range(n_resamples)]
    extreme = valid = 0
    for flips in patterns:
        diff = difference(metric, *swapped(pairs, flips))
        if diff is None:
            continue
        valid += 1
        extreme += abs(diff) >= abs(observed) - 1e-12
    if not valid:
        return PermutationResult(observed, None, 0, exact)
    p_value = extreme / valid if exact else (1 + extreme) / (1 + valid)
    return PermutationResult(observed, p_value, valid, exact)


# --- Holm correction --------------------------------------------------------------------


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    """Holm-adjusted p-values (step-down, monotone, capped at 1). None stays None and does
    not count towards the number of comparisons."""
    known = sorted((p, name) for name, p in p_values.items() if p is not None)
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    for rank, (p, name) in enumerate(known):
        running = max(running, min(1.0, (len(known) - rank) * p))
        adjusted[name] = running
    return adjusted


# --- Bootstrap by situation -------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    estimate: float | None
    low: float | None
    high: float | None
    n_resamples: int  # resamples with a defined metric


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics."""
    position = q * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_ci(
    records: Sequence[CaseRecord],
    metric: Metric,
    n_resamples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    unit: Unit = by_situation,
) -> Interval:
    """Percentile interval. A unit drawn twice gets a fresh situation id per draw, so
    tone sensitivity treats the copies as different situations."""
    units = [records for _, records in sorted(group_units(records, unit).items())]
    estimate = metric(records)
    rng = random.Random(seed)
    values = []
    for _ in range(n_resamples):
        sample = []
        for draw in range(len(units)):
            for r in rng.choice(units):
                sample.append(replace(r, situation_id=f"{r.situation_id}#{draw}"))
        value = metric(sample)
        if value is not None:
            values.append(value)
    if not values:
        return Interval(estimate, None, None, 0)
    values.sort()
    tail = (1 - level) / 2
    return Interval(estimate, quantile(values, tail), quantile(values, 1 - tail), len(values))


# --- Loading runs and comparing them ----------------------------------------------------


def load_records(
    run_dir: Path, cases_dir: Path | None = None, allow_test_split: bool = False
) -> list[CaseRecord]:
    """CaseRecords of a finished run, from its predictions and the cases it ran on."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cases = {c.case_id: c for c in load_cases(cases_dir or Path(config["cases"]), allow_test_split)}
    records = []
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    for line in filter(None, lines):
        p = json.loads(line)
        first, second = (
            Decision.model_validate(d) if d else None for d in (p["first"], p["second"])
        )
        records.append(CaseRecord.from_case(cases[p["case_id"]], first, second, p["followup_run"]))
    return records


def compare(
    system: Sequence[CaseRecord],
    others: dict[str, Sequence[CaseRecord]],
    metrics: Sequence[str] = PRE_REGISTERED,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """The system against each other run on each metric; Holm across all comparisons."""
    results: dict[str, dict[str, Any]] = {}
    for (other, records), name in itertools.product(others.items(), metrics):
        test = paired_permutation_test(system, records, METRICS[name], n_resamples, seed)
        results[f"{name}: system vs {other}"] = {
            **asdict(test),
            "system_ci": asdict(bootstrap_ci(system, METRICS[name], n_resamples, seed)),
            "other_ci": asdict(bootstrap_ci(records, METRICS[name], n_resamples, seed)),
        }
    adjusted = holm({key: r["p_value"] for key, r in results.items()})
    for key, result in results.items():
        result["p_holm"] = adjusted[key]
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.stats")
    parser.add_argument("system", type=Path, help="run dir of the system")
    parser.add_argument("others", type=Path, nargs="+", help="run dirs to compare with")
    parser.add_argument("--metrics", nargs="+", default=list(PRE_REGISTERED), choices=METRICS)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-test-split", action="store_true")
    args = parser.parse_args(argv)

    def records(path: Path) -> list[CaseRecord]:
        return load_records(path, allow_test_split=args.allow_test_split)

    others = {path.name: records(path) for path in args.others}
    results = compare(records(args.system), others, args.metrics, args.resamples, args.seed)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
