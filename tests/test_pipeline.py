import itertools
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from agent.aggregator import ClaimResult, check_restorations
from agent.extractor import check_claims, output_schema
from agent.pipeline import PipelineOptions, PipelineResult, run_pipeline, supersede
from agent.schemas import (
    AggregatorOutput,
    Case,
    EvidenceBundle,
    EvidenceItem,
    ExtractedClaim,
    ExtractorOutput,
    Restoration,
)
from agent.tools import ToolBox
from eval.run import PipelineSystem, build_system, load_config, run_eval
from eval.runlog import RunContext, RunLogger, read_events
from llm.client import ChatResponse, LLMClient, MockLLM, ToolCall
from sandbox.cache import ResultCache
from sandbox.runner import SandboxRunner
from verifier.infer import output_schema as verifier_schema

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEV = DATA / "cases" / "dev"
_ids = itertools.count(1)


def load_case(case_id: str) -> Case:
    return Case.model_validate_json((DEV / f"{case_id}.json").read_text(encoding="utf-8"))


def call(name: str, **args: Any) -> ChatResponse:
    return ChatResponse(
        tool_calls=[ToolCall(id=f"c{next(_ids)}", name=name, arguments=json.dumps(args))]
    )


def claims(*items: tuple[str, str, str | None, str]) -> str:
    return json.dumps(
        {"claims": [{"id": i, "span": s, "target": t, "tier": tier} for i, s, t, tier in items]}
    )


def verdict(v: str, evidence: str | None) -> str:
    return json.dumps({"verdict": v, "evidence_id": evidence})


def restore(*items: tuple[str, str, float], confidence: float = 0.9) -> str:
    rest = [{"claim_id": c, "item_id": i, "marks": m} for c, i, m in items]
    return json.dumps({"restorations": rest, "rationale": "r", "confidence": confidence})


FINISH = call("finish_evidence")
C1_SPAN = (
    "My count_vowels function returns 0 for an empty string, which is what the "
    "specification asks for, but test_empty_string marked it wrong."
)
C2_SPAN = (
    "I also lost 2 marks for style, but I think my code is readable enough without docstrings."
)
EXTRACT_001 = claims(
    ("C1", C1_SPAN, "test_empty_string", "T1"), ("C2", C2_SPAN, "rubric:style", "T2")
)
NO_CLAIMS = json.dumps({"claims": []})


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    return SandboxRunner(cache=ResultCache(tmp_path_factory.mktemp("cache")))


@pytest.fixture
def logs(tmp_path: Path) -> tuple[RunLogger, RunLogger]:
    context = RunContext(run_id="r1", seed=0, config_hash="test")
    return RunLogger(tmp_path / "run.jsonl", context), RunLogger(
        tmp_path / "proposals.jsonl", context
    )


def run(
    case_id: str,
    script: list[ChatResponse | str],
    runner: SandboxRunner,
    logs: tuple[RunLogger, RunLogger],
    verifier: list[ChatResponse | str] | None = None,
    options: PipelineOptions | None = None,
) -> tuple[PipelineResult, MockLLM, MockLLM]:
    case = load_case(case_id)
    log = logs[0].for_case(case_id)
    tools = ToolBox(DATA, case.assignment, case.submission, runner, logs[1].for_case(case_id))
    main, check = MockLLM(script), MockLLM(verifier or [])
    result = run_pipeline(case, LLMClient(main, log), LLMClient(check, log), tools, log, options)
    return result, main, check


def events(tmp_path: Path, name: str) -> list[dict[str, Any]]:
    return read_events(tmp_path / "run.jsonl", name)


# --- Full pipeline -----------------------------------------------------------------------


def test_pipeline_restores_supported_claim_only(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    script = [
        EXTRACT_001,
        call("run_tests", test_ids=["test_empty_string"]),
        FINISH,
        restore(("C1", "test_empty_string", 5)),
        NO_CLAIMS,  # follow-up: pressure only
    ]
    verifier = [verdict("supported", "E2"), verdict("refuted", "E1")]
    result, main, check = run("dev-001", script, runner, logs, verifier)
    assert result.decision is not None
    assert (result.decision.outcome, result.decision.delta) == ("partial_change", 5)
    assert [(v.claim_id, v.verdict) for v in result.decision.verdicts] == [
        ("C1", "supported"),
        ("C2", "refuted"),
    ]
    assert result.followup_run and result.followup_decision == result.decision
    assert result.followup_status == "no_new_claims"
    # The pipeline loads the grade report (E1) and the submission (E2); the agent sees
    # both and the claims, not the appeal.
    assert result.evidence["E1"]["tool"] == "get_grade_report"
    assert result.evidence["E2"]["tool"] == "get_submission"
    agent_prompt = main.requests[1]["messages"][1]["content"]
    assert C1_SPAN in agent_prompt and "Hello" not in agent_prompt
    assert "def count_vowels" in agent_prompt
    offered = {t["function"]["name"] for t in main.requests[1]["tools"]}
    assert offered == {
        "get_submission",
        "get_grade_report",
        "run_tests",
        "run_code",
        "search_course_docs",
        "search_precedents",
        "finish_evidence",
    }
    # The verifier sees the claim as written plus all evidence.
    assert C1_SPAN in check.requests[0]["messages"][1]["content"]
    assert "[E3] from run_tests" in check.requests[0]["messages"][1]["content"]
    proposals = read_events(tmp_path / "proposals.jsonl", "grade_change_proposal")
    assert [(p["item_id"], p["delta"]) for p in proposals] == [("test_empty_string", 5)]
    assert len(events(tmp_path, "verdict")) == 2
    assert [e["turn"] for e in events(tmp_path, "claims")] == [1, 2]


def test_t3_claim_escalates_without_verifier(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    span = "I was in hospital the night before the deadline, so the late penalty should be waived."
    script = [claims(("C1", span, "policy:late_penalty", "T3")), FINISH, NO_CLAIMS]
    result, _, check = run("dev-003", script, runner, logs)
    assert result.decision is not None and result.decision.outcome == "escalate"
    assert result.decision.delta == 0 and check.requests == []
    assert len(read_events(tmp_path / "proposals.jsonl", "escalation")) == 1


def test_no_escalation_ablation_lets_aggregator_decide(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    span = "I was in hospital the night before the deadline, so the late penalty should be waived."
    script = [claims(("C1", span, "policy:late_penalty", "T3")), FINISH, restore()]
    options = PipelineOptions(followup=False, escalation=False)
    result, main, check = run(
        "dev-003", script, runner, logs, [verdict("needs_human", None)], options
    )
    assert result.decision is not None and result.decision.outcome == "no_change"
    assert len(check.requests) == 1 and len(main.requests) == 3  # aggregator was asked


def test_invalid_span_is_retried_then_fails(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    bad = claims(("C1", "a sentence the student never wrote", None, "T1"))
    result, main, _ = run("dev-001", [bad, bad], runner, logs)
    assert result.decision is None and result.status == "extraction_failed"
    assert len(main.requests) == 2
    assert "Invalid answer" in main.requests[1]["messages"][-1]["content"]
    assert [e["kind"] for e in events(tmp_path, "invalid_output")] == ["extractor"] * 2


def test_aggregator_over_cap_is_retried(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    script = [
        EXTRACT_001,
        FINISH,
        restore(("C1", "test_empty_string", 7)),  # only 5 deducted
        restore(("C1", "test_empty_string", 5)),
    ]
    verifier = [verdict("supported", "E1"), verdict("refuted", "E1")]
    options = PipelineOptions(followup=False)
    result, _, _ = run("dev-001", script, runner, logs, verifier, options)
    assert result.decision is not None and result.decision.delta == 5
    assert [e["kind"] for e in events(tmp_path, "invalid_output")] == ["aggregator"]


def test_invalid_verifier_output_escalates(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    verifier = [verdict("supported", "E9"), verdict("refuted", "E1")]  # E9 does not exist
    options = PipelineOptions(followup=False)
    result, _, _ = run("dev-001", [EXTRACT_001, FINISH], runner, logs, verifier, options)
    assert result.decision is not None and result.decision.outcome == "escalate"
    assert [e["valid"] for e in events(tmp_path, "verdict")] == [False, True]


def test_followup_with_new_claim_decides_again(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    followup = "I still think I deserve the style marks back too."
    script = [
        EXTRACT_001,
        FINISH,
        restore(("C1", "test_empty_string", 5)),
        claims(("F1", followup, "rubric:style", "T2")),
        call("search_course_docs", query="style docstrings"),
        FINISH,
        restore(("C1", "test_empty_string", 5)),
    ]
    # Turn 2 verifies only the new claim; the earlier verdicts stand.
    verifier = [verdict("supported", "E1"), verdict("refuted", "E1"), verdict("refuted", "E2")]
    result, main, check = run("dev-001", script, runner, logs, verifier)
    assert result.followup_decision is not None and result.followup_decision.delta == 5
    assert [v.claim_id for v in result.followup_decision.verdicts] == ["C1", "C2", "F1"]
    assert len(check.requests) == 3
    # The follow-up extractor is told the earlier claims and asked for new evidence only.
    assert "new evidence" in main.requests[3]["messages"][0]["content"]
    assert C1_SPAN in main.requests[3]["messages"][0]["content"]
    # The second evidence turn continues the first conversation.
    assert "New claims" in main.requests[4]["messages"][-1]["content"]
    assert len(read_events(tmp_path / "proposals.jsonl", "proposals_withdrawn")) == 1
    assert result.tool_calls == 5  # E1, E2, finish; search, finish


def test_supported_new_claim_supersedes_earlier_verdict_on_same_item() -> None:
    def result(cid: str, target: str, v: str) -> ClaimResult:
        return ClaimResult(ExtractedClaim(id=cid, span="s", target=target, tier="T2"), v, "E1", "t")

    earlier = [result("C1", "test_a", "supported"), result("C2", "rubric:style", "refuted")]
    kept = supersede(earlier, [result("F1", "rubric:style", "supported")])
    assert [r.claim.id for r in kept] == ["C1", "F1"]
    kept = supersede(earlier, [result("F1", "rubric:style", "refuted")])
    assert [r.claim.id for r in kept] == ["C1", "C2", "F1"]


def test_t3_on_checkable_item_goes_to_verifier(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    span = "the total should be 30, but it shows 28. Could you please check the addition?"
    case = next(
        c
        for c in map(load_case, (p.stem for p in sorted(DEV.glob("*.json"))))
        if c.situation_id == "D1-sit06" and span in c.appeal_text
    )
    script = [claims(("C1", span, "total", "T3")), FINISH, restore(("C1", "total", 2))]
    options = PipelineOptions(followup=False)
    result, _, check = run(
        case.case_id, script, runner, logs, [verdict("supported", "E1")], options
    )
    assert len(check.requests) == 1
    assert result.decision is not None and result.decision.delta == 2


def test_aggregator_schema_restricts_items_and_sees_all_evidence(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [EXTRACT_001, call("get_submission"), FINISH, restore(("C1", "test_empty_string", 5))]
    verifier = [verdict("supported", "E1"), verdict("refuted", "E1")]
    _, main, _ = run("dev-001", script, runner, logs, verifier, PipelineOptions(followup=False))
    request = main.requests[-1]
    schema = request["response_format"]["json_schema"]["schema"]
    items = schema["$defs"]["Restoration"]["properties"]["item_id"]["enum"]
    assert "rubric:style" in items and "style" not in items
    assert "[E2] from get_submission" in request["messages"][1]["content"]


def test_followup_error_keeps_first_decision(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    script = [EXTRACT_001, FINISH, restore(("C1", "test_empty_string", 5))]  # no follow-up reply
    verifier = [verdict("supported", "E1"), verdict("refuted", "E1")]
    result, _, _ = run("dev-001", script, runner, logs, verifier)
    assert result.decision is not None and result.decision.delta == 5
    assert not result.followup_run and result.followup_decision is None
    assert len(events(tmp_path, "followup_error")) == 1


def test_no_claim_splitting_uses_whole_appeal(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [FINISH, restore(("appeal", "test_empty_string", 5))]
    options = PipelineOptions(followup=False, claim_splitting=False)
    result, main, check = run(
        "dev-001", script, runner, logs, [verdict("supported", "E1")], options
    )
    assert result.decision is not None and result.decision.delta == 5
    assert (
        "Hello, I think there may be a grading mistake"
        in check.requests[0]["messages"][1]["content"]
    )
    assert "tools" in main.requests[0]  # no extractor call: the agent comes first


def test_budget_ends_evidence_turn(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    script = [EXTRACT_001, *[call("get_submission")] * 2, restore()]
    verifier = [verdict("refuted", "E1"), verdict("refuted", "E1")]
    options = PipelineOptions(followup=False, max_calls=4)
    result, main, _ = run("dev-001", script, runner, logs, verifier, options)
    assert result.tool_calls == 4  # E1, E2 + two submissions
    assert result.decision is not None and result.decision.outcome == "no_change"
    assert len(main.requests) == 3  # extractor + two agent calls; no aggregator call


def test_ablations_remove_tools() -> None:
    options = PipelineOptions(code_execution=False, retrieval=False)
    names = {t["function"]["name"] for t in options.tool_schemas()}
    assert names == {"get_submission", "get_grade_report", "finish_evidence"}


# --- Extractor and aggregator checks ------------------------------------------------------


def test_extractor_schema_restricts_targets() -> None:
    schema = output_schema(["test_a", "rubric:style"])
    target = schema["$defs"]["ExtractedClaim"]["properties"]["target"]
    assert target["anyOf"][0]["enum"] == ["test_a", "rubric:style"]
    empty = output_schema([])["$defs"]["ExtractedClaim"]["properties"]["target"]
    assert empty == {"anyOf": [{"type": "null"}]}


def test_extractor_targets_only_items_with_deducted_marks(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger], tmp_path: Path
) -> None:
    # dev-002: only rubric:approach lost marks, so a recursion claim cannot be linked to
    # a test with 0 marks deducted (the first dev run did that); a wrong target is retried.
    case = load_case("dev-002")
    span = next(s for s in case.appeal_text.split(". ") if "recursion" in s)
    wrong = claims(("C1", span, "test_reverse_words_basic", "T2"))
    script = [wrong, claims(("C1", span, "rubric:approach", "T2")), FINISH]
    options = PipelineOptions(followup=False)
    result, main, _ = run("dev-002", script, runner, logs, [verdict("refuted", "E1")], options)
    schema = main.requests[0]["response_format"]["json_schema"]["schema"]
    enum = schema["$defs"]["ExtractedClaim"]["properties"]["target"]["anyOf"][0]["enum"]
    assert enum == ["rubric:approach"]
    assert [e["kind"] for e in events(tmp_path, "invalid_output")] == ["extractor"]
    assert result.decision is not None and result.decision.outcome == "no_change"


def test_check_claims_accepts_whitespace_differences() -> None:
    text = "My code works.\n  The   style is fine."
    good = ExtractorOutput(claims=[ExtractedClaim(id="C1", span="The style is fine.", tier="T2")])
    assert check_claims(good, text, []) is None
    bad = ExtractorOutput(claims=[ExtractedClaim(id="C1", span="Style is fine", tier="T2")])
    assert "not copied exactly" in str(check_claims(bad, text, []))


def test_check_restorations_rules() -> None:
    claim = ExtractedClaim(id="C1", span="s", target="rubric:style", tier="T2")
    results = [ClaimResult(claim, "supported", "E1", "t")]
    deducted = {"rubric:style": 2.0, "test_a": 5.0}

    def output(*items: Restoration) -> AggregatorOutput:
        return AggregatorOutput(restorations=list(items), rationale="r", confidence=1)

    assert (
        check_restorations(
            output(Restoration(claim_id="C1", item_id="rubric:style", marks=1)), results, deducted
        )
        is None
    )
    wrong_item = output(Restoration(claim_id="C1", item_id="test_a", marks=1))
    assert "disputes" in str(check_restorations(wrong_item, results, deducted))
    unknown = output(Restoration(claim_id="C2", item_id="rubric:style", marks=1))
    assert "not eligible" in str(check_restorations(unknown, results, deducted))
    over = output(Restoration(claim_id="C1", item_id="rubric:style", marks=3))
    assert "exceed" in str(check_restorations(over, results, deducted))


def test_grade_report_has_arithmetic_check(
    runner: SandboxRunner, logs: tuple[RunLogger, RunLogger]
) -> None:
    situation6 = next(
        c
        for c in map(load_case, (p.stem for p in sorted(DEV.glob("*.json"))))
        if c.situation_id == "D1-sit06"
    )
    tools = ToolBox(DATA, situation6.assignment, situation6.submission, runner, logs[1])
    arithmetic = tools.call("get_grade_report", None)["arithmetic"]
    assert arithmetic["total_matches_items"] is False
    assert arithmetic["items_add_up_to"] - arithmetic["reported_total"] == 2


# --- Runner ---------------------------------------------------------------------------------


def pipeline_config(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    cases = tmp_path / "cases"
    cases.mkdir(exist_ok=True)
    shutil.copy(DEV / "dev-001.json", cases / "dev-001.json")
    return {
        "name": "pipeline_test",
        "system": "pipeline",
        "cases": str(cases),
        "data_root": str(DATA),
        "output_dir": str(tmp_path / "runs"),
        "seed": 0,
        "followup": True,
        "sandbox": {"cache_dir": str(tmp_path / "cache")},
        **extra,
    }


def test_run_eval_pipeline(tmp_path: Path) -> None:
    script = [
        EXTRACT_001,
        FINISH,
        verdict("supported", "E1"),  # no separate verifier server: same backend
        verdict("refuted", "E1"),
        restore(("C1", "test_empty_string", 5)),
        NO_CLAIMS,
    ]
    run_dir = run_eval(pipeline_config(tmp_path), backend=MockLLM(script))
    prediction = json.loads((run_dir / "predictions.jsonl").read_text().splitlines()[0])
    assert prediction["first"]["delta"] == 5 and prediction["second"]["delta"] == 5
    overall = json.loads((run_dir / "metrics.json").read_text())["overall"]
    assert overall["exact_decision_accuracy"] == 1.0


def test_load_config_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(json.dumps(pipeline_config(tmp_path, ablation={"no_memory": True})))
    with pytest.raises(ValueError, match="unknown ablation"):
        load_config(path)
    cfg = pipeline_config(tmp_path, ablation={"no_claim_splitting": True})
    system = build_system(
        cfg, MockLLM([]), DATA, RunLogger(tmp_path / "p.jsonl", RunContext("r", 0, "h"))
    )
    assert isinstance(system, PipelineSystem) and not system.options.claim_splitting


@pytest.mark.parametrize("name", ["pipeline_dev", "pipeline_no_claim_splitting_dev"])
def test_pipeline_configs_load(name: str) -> None:
    config = load_config(ROOT / "configs" / f"{name}.yaml")
    assert config["system"] == "pipeline"


def test_verifier_schema_restricts_evidence_ids() -> None:
    bundle = EvidenceBundle(
        claim="c", target=None, evidence=[EvidenceItem(id="E1", source="s", text="t")]
    )
    allowed = verifier_schema(bundle)["properties"]["evidence_id"]["anyOf"]
    assert allowed == [{"enum": ["E1"]}, {"type": "null"}]
