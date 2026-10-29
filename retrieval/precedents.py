"""Precedent store: past appeal decisions the agent can look up with `search_precedents`.

In the main evaluation the store holds a few verified decisions per assignment, written by
the team on grading situations that are not in the evaluated set (proposal section 7); in
the precedent runs for RQ2 it fills with the agent's own decisions. Stored as JSON lines,
one `Precedent` per line. Search is BM25 over the issue, target and rationale, restricted
to the same assignment.
"""

from pathlib import Path

from pydantic import Field
from rank_bm25 import BM25Plus

from agent.schemas import NonEmptyStr, Outcome, StrictModel
from retrieval.course_docs import tokenize


class Precedent(StrictModel):
    precedent_id: str = NonEmptyStr
    assignment: str = NonEmptyStr
    target: str = NonEmptyStr  # test id, rubric:..., policy:... or total
    issue: str = NonEmptyStr  # what the student disputed, in neutral words
    outcome: Outcome
    delta: float = Field(ge=0)
    rationale: str = NonEmptyStr

    def text(self) -> str:
        return f"{self.target} {self.issue} {self.rationale}"


class PrecedentStore:
    def __init__(self, precedents: list[Precedent] | None = None) -> None:
        self.precedents: list[Precedent] = list(precedents or [])

    @classmethod
    def load(cls, path: Path) -> "PrecedentStore":
        lines = path.read_text(encoding="utf-8").splitlines()
        return cls([Precedent.model_validate_json(line) for line in lines if line.strip()])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [p.model_dump_json() for p in self.precedents]
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

    def add(self, precedent: Precedent) -> None:
        if any(p.precedent_id == precedent.precedent_id for p in self.precedents):
            raise ValueError(f"duplicate precedent id {precedent.precedent_id!r}")
        self.precedents.append(precedent)

    def search(self, query: str, assignment: str, k: int = 3) -> list[Precedent]:
        """Up to k precedents of this assignment sharing at least one word with the query,
        best BM25 match first. BM25Plus keeps scores positive even in a tiny store."""
        pool = [p for p in self.precedents if p.assignment == assignment]
        terms = set(tokenize(query))
        if not pool or not terms:
            return []
        docs = [tokenize(p.text()) for p in pool]
        scores = BM25Plus(docs).get_scores(list(terms))
        matching = [i for i, doc in enumerate(docs) if terms & set(doc)]
        ranked = sorted(matching, key=lambda i: (-scores[i], i))
        return [pool[i] for i in ranked[:k]]
