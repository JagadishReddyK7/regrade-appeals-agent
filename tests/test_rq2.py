"""RQ2: consistency cases (data/build_rq2.py), consistency metrics (eval/metrics.py),
precedent runs (eval/run.py) and the opt-in precedent preload (agent/pipeline.py)."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent.pipeline import PipelineOptions, run_pipeline
from agent.schemas import Case
from agent.tools import ToolBox
from data.build_cases import BuildError, case_json
from data.build_rq2 import build_set, load_set
from eval.metrics import (
    CaseRecord,
    Prediction,
    compute_consistency,
    group_consistency,
    group_mark_gap,
    group_split,
    pair_flip_rate,
    precedent_copy_rate,
)
from eval.run import ordered_cases, planted_copy_rate, precedent_from, run_eval
from eval.runlog import RunContext, RunLogger, read_events
from llm.client import LLMClient, MockLLM
from retrieval.precedents import Precedent, PrecedentStore
from sandbox.cache import ResultCache
from sandbox.runner import SandboxRunner
from tests.test_pipeline import EXTRACT_001, FINISH, NO_CLAIMS, restore, verdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SETS = sorted((DATA / "rq2").glob("*.yaml"))
RQ2 = DATA / "cases" / "rq2"


def load(folder: Path, case_id: str) -> Case:
    return Case.model_validate_json((folder / f"{case_id}.json").read_text(encoding="utf-8"))


# --- Builder ------------------------------------------------------------------------------


def test_committed_rq2_cases_match_their_set_files() -> None:
    built = {case.case_id: case_json(case) for path in SETS for case in build_set(load_set(path))}
    on_disk = {p.stem: p.read_text(encoding="utf-8") for p in RQ2.glob("*.json")}
    assert len(built) == 32 and built == on_disk


def test_groups_and_pairs_get_their_ids() -> None:
    group = build_set(load_set(DATA / "rq2" / "D1-g1.yaml"))
    assert {c.group_id for c in group} == {"D1-g1"} and {c.pair_id for c in group} == {None}
    assert len({c.submission for c in group}) == 4
    assert [c.situation_id for c in group] == [f"D1-g1-{n}" for n in range(1, 5)]
    pair = build_set(load_set(DATA / "rq2" / "D1-p1.yaml"))
    assert {c.pair_id for c in pair} == {"D1-p1"} and len({c.submission for c in pair}) == 1


def test_member_spans_must_cover_the_claims() -> None:
    cset = load_set(DATA / "rq2" / "D1-g1.yaml")
    member = cset.members[0].model_copy(update={"spans": {}})
    broken = cset.model_copy(update={"members": [member, *cset.members[1:]]})
    with pytest.raises(BuildError, match="spans must cover"):
        build_set(broken)


# --- Consistency metrics ------------------------------------------------------------------

NO, FIVE, THREE = (
    Prediction("no_change", 0),
    Prediction("full_change", 5),
    Prediction("full_change", 3),
)


def rec(
    gold: Prediction,
    first: Prediction | None,
    group: str | None = None,
    pair: str | None = None,
    cid: str = "c",
) -> CaseRecord:
    return CaseRecord(
        case_id=cid,
        situation_id=cid,
        tone="polite",
        gold=gold,
        gold_after=gold,
        has_new_evidence=False,
        first=first,
        group_id=group,
        pair_id=pair,
    )


def test_group_consistency_and_three_way_split() -> None:
    records = [
        rec(FIVE, FIVE, "g1"),
        rec(FIVE, FIVE, "g1"),  # consistent and correct
        rec(FIVE, NO, "g2"),
        rec(FIVE, NO, "g2"),  # consistent but wrong
        rec(FIVE, FIVE, "g3"),
        rec(FIVE, None, "g3"),  # inconsistent (no decision differs)
        rec(FIVE, FIVE, "lonely"),  # one member: not a group
    ]
    assert group_consistency(records) == pytest.approx(2 / 3)
    assert group_split(records) == pytest.approx(
        {"consistent_correct": 1 / 3, "consistent_wrong": 1 / 3, "inconsistent": 1 / 3}
    )


def test_group_mark_gap_uses_decided_members_only() -> None:
    records = [
        rec(FIVE, FIVE, "g1"),
        rec(FIVE, THREE, "g1"),  # gaps 0 and 2: mean 1
        rec(FIVE, NO, "g2"),
        rec(FIVE, Prediction("escalate", 0), "g2"),  # gap 5; escalation out
        rec(FIVE, None, "g3"),
        rec(FIVE, None, "g3"),  # nothing decided: left out
    ]
    assert group_mark_gap(records) == pytest.approx(3.0)


def test_pair_flip_rate() -> None:
    records = [
        rec(FIVE, FIVE, pair="p1"),
        rec(FIVE, NO, pair="p1"),
        rec(NO, NO, pair="p2"),
        rec(NO, NO, pair="p2"),
    ]
    assert pair_flip_rate(records) == 0.5


def test_precedent_copy_rate_counts_repeats_of_earlier_wrong_decisions() -> None:
    records = [
        rec(FIVE, FIVE, "g1"),  # correct: nothing to copy yet
        rec(FIVE, NO, "g1"),  # first wrong decision
        rec(FIVE, NO, "g1"),  # copies it
        rec(FIVE, FIVE, "g1"),  # does not
    ]
    assert precedent_copy_rate(records) == 0.5
    # Order matters: wrong first, then two correct members, then a repeat of the error.
    reordered = [records[1], records[0], records[3], records[2]]
    assert precedent_copy_rate(reordered) == pytest.approx(1 / 3)


def test_compute_consistency_without_sets_is_empty() -> None:
    result = compute_consistency([rec(FIVE, FIVE)])
    assert result["n_groups"] == 0 and result["group_consistency"] is None


# --- Precedent runs -----------------------------------------------------------------------


def test_case_order_is_shuffled_reproducibly() -> None:
    cases = [load(RQ2, f"rq2-{n:03d}") for n in range(1, 9)]
    assert ordered_cases(cases, {}) == cases
    shuffled = ordered_cases(cases, {"case_order_seed": 1})
    assert shuffled == ordered_cases(cases, {"case_order_seed": 1})
    assert shuffled != cases and sorted(c.case_id for c in shuffled) == [c.case_id for c in cases]


def test_precedent_is_built_from_the_decision_not_the_answer_key() -> None:
    from agent.schemas import ClaimVerdict, Decision
    from baselines.base import SystemOutput

    case = load(RQ2, "rq2-001")  # correct answer: full_change +5
    decision = Decision(
        case_id=case.case_id,
        outcome="no_change",
        delta=0,
        rationale="test ok",
        confidence=0.8,
        verdicts=[ClaimVerdict(claim_id="C1", verdict="refuted", evidence_ids=["E1"])],
    )
    precedent = precedent_from(case, SystemOutput(first=decision))
    assert precedent is not None
    assert (precedent.outcome, precedent.delta, precedent.issue) == (
        "no_change",
        0,
        case.appeal_text,
    )
    assert precedent_from(case, SystemOutput(first=None)) is None


def test_planted_copy_rate() -> None:
    cases = [load(RQ2, "rq2-001"), load(RQ2, "rq2-002"), load(RQ2, "rq2-013")]
    planted = [
        Precedent(
            precedent_id="P",
            assignment="D1",
            target="test_empty_string",
            issue="empty string test",
            outcome="no_change",
            delta=0,
            rationale="the autograder is authoritative",
        )
    ]
    records = [
        CaseRecord.from_case(cases[0], None),
        CaseRecord.from_case(cases[1], None),
        CaseRecord.from_case(cases[2], None),
    ]
    records[0] = rec(FIVE, NO)  # copies the planted decision
    records[1] = rec(FIVE, FIVE)  # does not
    # rq2-013 disputes another test: not eligible.
    assert planted_copy_rate(cases, records, planted) == 0.5


def test_planted_follow_rate_counts_a_correct_precedent() -> None:
    """A correct (helpful) precedent has no copies of a wrong decision; the follow rate
    counts every case on its target, so it is the accuracy on those cases."""
    cases = [load(RQ2, "rq2-001"), load(RQ2, "rq2-002"), load(RQ2, "rq2-013")]
    helpful = [
        Precedent(
            precedent_id="H",
            assignment="D1",
            target="test_empty_string",
            issue="empty string test",
            outcome="full_change",
            delta=5,
            rationale="the test is wrong",
        )
    ]
    records = [rec(FIVE, FIVE), rec(FIVE, NO), rec(NO, FIVE)]
    assert planted_copy_rate(cases, records, helpful) is None  # no case where it is wrong
    assert planted_copy_rate(cases, records, helpful, only_wrong=False) == 0.5


# --- Precedent preload in the pipeline ----------------------------------------------------


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    return SandboxRunner(cache=ResultCache(tmp_path_factory.mktemp("cache")))


def run_dev_001(
    runner: SandboxRunner, tmp_path: Path, options: PipelineOptions, store: PrecedentStore
) -> tuple[Any, MockLLM]:
    case = load(DATA / "cases" / "dev", "dev-001")
    context = RunContext(run_id="r1", seed=0, config_hash="test")
    log = RunLogger(tmp_path / "run.jsonl", context).for_case(case.case_id)
    proposals = RunLogger(tmp_path / "proposals.jsonl", context).for_case(case.case_id)
    tools = ToolBox(DATA, case.assignment, case.submission, runner, proposals, precedents=store)
    main = MockLLM([EXTRACT_001, FINISH, restore(("C1", "test_empty_string", 5)), NO_CLAIMS])
    check = MockLLM([verdict("supported", "E1"), verdict("refuted", "E1")])
    result = run_pipeline(case, LLMClient(main, log), LLMClient(check, log), tools, log, options)
    return result, main


def test_preload_precedents_is_off_by_default_and_opt_in(
    runner: SandboxRunner, tmp_path: Path
) -> None:
    assert PipelineOptions.from_config({}).preload_precedents == 0
    store = PrecedentStore(
        [
            Precedent(
                precedent_id="P",
                assignment="D1",
                target="appeal",
                issue="count_vowels returns 0 for an empty string",
                outcome="full_change",
                delta=5,
                rationale="faulty test",
            )
        ]
    )
    off, _ = run_dev_001(runner, tmp_path / "off", PipelineOptions(), store)
    assert [e["tool"] for e in off.evidence.values()] == ["get_grade_report", "get_submission"]
    options = PipelineOptions.from_config({"retrieval": {"preload_precedents": 2}})
    on, main = run_dev_001(runner, tmp_path / "on", options, store)
    assert on.evidence["E3"]["tool"] == "search_precedents"
    assert on.evidence["E3"]["output"]["precedents"][0]["precedent_id"] == "P"
    agent_prompt = main.requests[1]["messages"][1]["content"]
    assert "faulty test" in agent_prompt and "Hello" not in agent_prompt  # not the appeal


def test_run_eval_accumulates_precedents(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    shutil.copy(DATA / "cases" / "dev" / "dev-001.json", cases / "dev-001.json")
    config: dict[str, Any] = {
        "name": "precedent_test",
        "system": "pipeline",
        "cases": str(cases),
        "data_root": str(DATA),
        "output_dir": str(tmp_path / "runs"),
        "seed": 0,
        "followup": False,
        "case_order_seed": 3,
        "sandbox": {"cache_dir": str(tmp_path / "c")},
        "retrieval": {"accumulate_precedents": True, "preload_precedents": 3},
    }
    script = [
        EXTRACT_001,
        FINISH,
        verdict("supported", "E1"),
        verdict("refuted", "E1"),
        restore(("C1", "test_empty_string", 5)),
    ]
    run_dir = run_eval(config, backend=MockLLM(script))
    added = read_events(run_dir / "events.jsonl", "precedent_added")
    assert [(e["precedent"]["precedent_id"], e["precedent"]["delta"]) for e in added] == [
        ("dev-001", 5)
    ]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "consistency" in metrics and "planted_copy_rate" not in metrics["consistency"]


# --- Configs ------------------------------------------------------------------------------

RQ2_CONFIGS = {
    "rq2_b1": "b1_dev",
    "rq2_agent": "agent_dev",
    "rq2_pipeline": "pipeline_ft_dev",
    "rq2_pipeline_precedents_s1": "pipeline_ft_dev",
    "rq2_pipeline_precedents_s2": "pipeline_ft_dev",
    "rq2_pipeline_precedents_s3": "pipeline_ft_dev",
    "rq2_pipeline_planted": "pipeline_ft_dev",
    "rq2_pipeline_r2": "pipeline_ft_dev",
    "rq2_pipeline_r3": "pipeline_ft_dev",
    "rq2_pipeline_planted_recursion": "pipeline_ft_dev",
    "rq2_pipeline_helpful": "pipeline_ft_dev",
}
RQ2_KEYS = {"name", "cases", "followup", "case_order_seed", "retrieval"}


@pytest.mark.parametrize("name", sorted(RQ2_CONFIGS))
def test_rq2_configs_equal_their_dev_configs(name: str) -> None:
    """Only the name, the case folder, the follow-up switch and the RQ2 settings differ, so
    no setting is tuned for RQ2 (the frozen system, docs/DECISIONS.md sections 13 and 16)."""

    def read(stem: str) -> dict[str, Any]:
        return yaml.safe_load((ROOT / "configs" / f"{stem}.yaml").read_text(encoding="utf-8"))

    rq2, dev = read(name), read(RQ2_CONFIGS[name])
    assert (rq2["name"], rq2["cases"], rq2["followup"]) == (name, "data/cases/rq2", False)
    strip = lambda c: {k: v for k, v in c.items() if k not in RQ2_KEYS}  # noqa: E731
    assert strip(rq2) == strip(dev)
    assert set(RQ2_CONFIGS) == {p.stem for p in (ROOT / "configs").glob("rq2_*.yaml")}
