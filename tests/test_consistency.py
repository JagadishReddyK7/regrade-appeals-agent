from pathlib import Path
from typing import Any

from agent.schemas import Case
from data.consistency import check_consistency
from tests.test_schemas import make_case


def build(cases: list[dict[str, Any]]) -> dict[Path, Case]:
    return {Path(f"{c['case_id']}.json"): Case.model_validate(c) for c in cases}


def same_submission(case: Case) -> str:
    return case.submission


def errors_for(cases: list[dict[str, Any]], base_of=same_submission) -> dict[str, list[str]]:
    found = check_consistency(build(cases), base_of)
    return {path.stem: messages for path, messages in found.items()}


def variant(case_id: str, tone: str, **overrides: Any) -> dict[str, Any]:
    return make_case(case_id=case_id, tone=tone, **overrides)


def test_consistent_situation_passes() -> None:
    cases = [
        variant("a", "polite"),
        variant("b", "authority", appeal_text=make_case()["appeal_text"]),
    ]
    assert errors_for(cases) == {}


def test_situation_with_different_answer_flagged() -> None:
    other = variant("b", "emotional", answer={"outcome": "partial_change", "delta": 3})
    other["claims"].append(
        {
            "id": "C2",
            "span": "please recheck",
            "tier": "T2",
            "target": "rubric:style",
            "disputed_marks": 2,
            "gold_verdict": "refuted",
        }
    )
    found = errors_for([variant("a", "polite"), other])
    assert "situation D1-sit01" in found["b"][0]


def test_situation_with_different_submission_flagged() -> None:
    found = errors_for([variant("a", "polite"), variant("b", "emotional", submission="D1-S09")])
    assert "b" in found


def test_hidden_instruction_variant_counts_as_same_submission() -> None:
    cases = [variant("a", "polite"), variant("b", "hidden_instruction", submission="D1-S01-hi")]
    assert errors_for(cases, base_of=lambda c: c.submission.removesuffix("-hi")) == {}


def test_duplicate_tone_in_situation_flagged() -> None:
    found = errors_for([variant("a", "polite"), variant("b", "polite")])
    assert any("appears more than once" in e for e in found["a"])


def test_equivalence_group_needs_same_answer_and_different_submissions() -> None:
    base = {"group_id": "g1"}
    ok = [
        make_case(case_id="a", situation_id="s1", submission="D1-S01", **base),
        make_case(case_id="b", situation_id="s2", submission="D1-S02", **base),
    ]
    assert errors_for(ok) == {}
    same_sub = [
        make_case(case_id="a", situation_id="s1", **base),
        make_case(case_id="b", situation_id="s2", **base),
    ]
    assert any("must all differ" in e for e in errors_for(same_sub)["a"])
    wrong = make_case(
        case_id="b",
        situation_id="s2",
        submission="D1-S02",
        planted_error=None,
        answer={"outcome": "no_change", "delta": 0},
        **base,
    )
    wrong["claims"][0]["gold_verdict"] = "refuted"
    found = errors_for([ok[0], wrong])
    assert "equivalence group g1" in found["b"][0]


def test_minimal_pair_needs_exactly_two() -> None:
    cases = [make_case(case_id=c, situation_id=f"s{c}", pair_id="p1", tone="polite") for c in "abc"]
    found = errors_for(cases)
    assert all(any("not 2" in e for e in found[c]) for c in "abc")
