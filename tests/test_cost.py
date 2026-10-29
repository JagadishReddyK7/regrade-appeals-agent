import pytest

from eval.cost import per_case_cost, summarise_cost

EVENTS = [
    {"event": "run_start", "case_id": None},
    {
        "event": "llm_call",
        "case_id": "a",
        "latency_s": 2.0,
        "prompt_tokens": 100,
        "completion_tokens": 10,
    },
    {
        "event": "llm_call",
        "case_id": "a",
        "latency_s": 1.0,
        "prompt_tokens": 50,
        "completion_tokens": 5,
    },
    {"event": "tool_call", "case_id": "a", "status": "ok"},
    {"event": "tool_call", "case_id": "a", "status": "invalid"},
    {"event": "tool_call", "case_id": "a", "status": "skipped"},  # not executed: not counted
    {
        "event": "llm_call",
        "case_id": "b",
        "latency_s": 4.0,
        "prompt_tokens": 200,
        "completion_tokens": 20,
    },
    {"event": "decision", "case_id": "b"},
]


def test_per_case_cost() -> None:
    costs = per_case_cost(EVENTS, {"a": 5.0, "b": 3.0, "c": 0.5})
    assert costs["a"] == {
        "wall_s": 5.0,
        "llm_calls": 2,
        "llm_latency_s": 3.0,
        "prompt_tokens": 150,
        "completion_tokens": 15,
        "tool_calls": 2,
    }
    assert costs["b"]["llm_calls"] == 1 and costs["b"]["tool_calls"] == 0
    assert costs["c"]["llm_calls"] == 0 and costs["c"]["wall_s"] == 0.5  # e.g. B0: no LLM


def test_summarise_cost() -> None:
    summary = summarise_cost(per_case_cost(EVENTS, {"a": 5.0, "b": 3.0}))
    assert summary["n_appeals"] == 2
    assert summary["mean_per_appeal"]["prompt_tokens"] == pytest.approx(175)
    assert summary["median_per_appeal"]["wall_s"] == pytest.approx(4.0)
    assert summary["total"]["llm_calls"] == 3


def test_summarise_cost_empty() -> None:
    assert summarise_cost({}) == {"n_appeals": 0}
