"""Cost per appeal, computed from a run's events: wall time, LLM calls, latency, tokens and
tool calls. Reported in metrics.json under `cost`; separate from the task metrics."""

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

COST_FIELDS = (
    "wall_s",
    "llm_calls",
    "llm_latency_s",
    "prompt_tokens",
    "completion_tokens",
    "tool_calls",
)


def per_case_cost(
    events: Iterable[Mapping[str, Any]], wall_s: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """Cost fields for every case in wall_s (cases without events get zeros)."""
    costs: dict[str, dict[str, float]] = defaultdict(lambda: dict.fromkeys(COST_FIELDS, 0.0))
    for event in events:
        case_id = event.get("case_id")
        if case_id is None:
            continue
        cost = costs[case_id]
        if event["event"] == "llm_call":
            cost["llm_calls"] += 1
            cost["llm_latency_s"] += event.get("latency_s") or 0.0
            cost["prompt_tokens"] += event.get("prompt_tokens") or 0
            cost["completion_tokens"] += event.get("completion_tokens") or 0
        elif event["event"] == "tool_call" and event.get("status") != "skipped":
            cost["tool_calls"] += 1
    for case_id, seconds in wall_s.items():
        costs[case_id]["wall_s"] = seconds
    return {case_id: costs[case_id] for case_id in wall_s}


def summarise_cost(per_case: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Mean, median and total of each cost field over the appeals."""
    if not per_case:
        return {"n_appeals": 0}
    columns = {name: [cost[name] for cost in per_case.values()] for name in COST_FIELDS}
    return {
        "n_appeals": len(per_case),
        "mean_per_appeal": {name: statistics.fmean(v) for name, v in columns.items()},
        "median_per_appeal": {name: statistics.median(v) for name, v in columns.items()},
        "total": {name: sum(v) for name, v in columns.items()},
    }
