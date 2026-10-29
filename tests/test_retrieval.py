from pathlib import Path

import pytest

from agent.schemas import SearchArgs
from eval.run import load_precedents
from retrieval.course_docs import CourseDoc, CourseDocIndex, in_force_first
from retrieval.precedents import Precedent, PrecedentStore

DATA = Path(__file__).resolve().parents[1] / "data"
D1 = DATA / "assignments" / "D1"


def ids(docs: list[CourseDoc]) -> list[str]:
    return [d.doc_id for d in docs]


# --- In-force ranking -------------------------------------------------------------------


def test_in_force_announcement_ranks_above_superseded() -> None:
    index = CourseDocIndex.for_assignment(D1)
    # The query uses A1's own wording, so BM25 alone would rank A1 first.
    found = ids(index.search("Clarification: no recursion anywhere in D1", k=5))
    assert "announcement:A1" in found
    assert found.index("announcement:A2") < found.index("announcement:A1")


def test_in_force_announcement_added_even_if_it_did_not_match() -> None:
    a1 = CourseDoc("announcement:A1", "announcement", "old", "old rule", "2026-09-01", "A2")
    a2 = CourseDoc("announcement:A2", "announcement", "new", "new rule", "2026-09-10", None)
    spec = CourseDoc("spec:0", "spec", "Spec", "text")
    assert ids(in_force_first([a1, spec], [a1, a2, spec])) == [
        "announcement:A2",
        "announcement:A1",
        "spec:0",
    ]


def test_supersession_chain_follows_to_newest() -> None:
    a1 = CourseDoc("announcement:A1", "announcement", "1", "x", "2026-09-01", "A2")
    a2 = CourseDoc("announcement:A2", "announcement", "2", "x", "2026-09-02", "A3")
    a3 = CourseDoc("announcement:A3", "announcement", "3", "x", "2026-09-03", None)
    ordered = ids(in_force_first([a1, a2], [a1, a2, a3]))
    assert ordered == ["announcement:A3", "announcement:A1", "announcement:A2"]


def test_search_respects_k_after_reordering() -> None:
    index = CourseDocIndex.for_assignment(D1)
    assert len(index.search("recursion reverse_words announcement", k=1)) == 1


# --- Precedent store --------------------------------------------------------------------


def precedent(pid: str, assignment: str = "D1", **kw: object) -> Precedent:
    fields = {
        "target": "rubric:approach",
        "issue": "student used recursion where the old rule forbade it",
        "outcome": "full_change",
        "delta": 3,
        "rationale": "the later announcement allows recursion",
        **kw,
    }
    return Precedent(precedent_id=pid, assignment=assignment, **fields)


@pytest.fixture
def store() -> PrecedentStore:
    return PrecedentStore(
        [
            precedent("p1"),
            precedent(
                "p2",
                target="test_empty_string",
                issue="test expects None for an empty string",
                rationale="the specification says 0; the test is wrong",
                outcome="partial_change",
                delta=5,
            ),
            precedent("p3", assignment="D2"),
        ]
    )


def test_search_finds_best_match_in_same_assignment(store: PrecedentStore) -> None:
    found = store.search("recursion approach marks", "D1")
    assert [p.precedent_id for p in found][0] == "p1"
    assert all(p.assignment == "D1" for p in found)


def test_search_needs_a_shared_word(store: PrecedentStore) -> None:
    assert store.search("late penalty hospital", "D1") == []
    assert store.search("", "D1") == []
    assert PrecedentStore().search("recursion", "D1") == []


def test_save_and_load_roundtrip(store: PrecedentStore, tmp_path: Path) -> None:
    path = tmp_path / "precedents.jsonl"
    store.save(path)
    assert PrecedentStore.load(path).precedents == store.precedents


def test_duplicate_precedent_id_rejected(store: PrecedentStore) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        store.add(precedent("p1"))


def test_runner_loads_precedents_from_config(store: PrecedentStore, tmp_path: Path) -> None:
    path = tmp_path / "precedents.jsonl"
    store.save(path)
    assert len(load_precedents({"retrieval": {"precedents": str(path)}}).precedents) == 3
    assert load_precedents({}).precedents == []


def test_search_precedents_tool(store: PrecedentStore, tmp_path: Path) -> None:
    from agent.tools import ToolBox
    from eval.runlog import RunContext, RunLogger
    from sandbox.runner import SandboxRunner

    log = RunLogger(tmp_path / "p.jsonl", RunContext("r", 0, "h"))
    tools = ToolBox(DATA, "D1", "D1-S11", SandboxRunner(), log, precedents=store)
    found = tools.call("search_precedents", SearchArgs(query="recursion approach"))
    assert [p["precedent_id"] for p in found["precedents"]] == ["p1"]
    empty = ToolBox(DATA, "D1", "D1-S11", SandboxRunner(), log)
    assert empty.call("search_precedents", SearchArgs(query="recursion"))["precedents"] == []
