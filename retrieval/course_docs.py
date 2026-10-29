"""BM25 search over an assignment's course documents: spec sections, rubric items and
dated announcements. Announcements carry their date and which later announcement
supersedes them, so the agent can tell which rule is in force."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from data.formats import load_announcements, load_assignment


@dataclass(frozen=True)
class CourseDoc:
    doc_id: str
    kind: str  # "spec", "rubric" or "announcement"
    title: str
    text: str
    date: str | None = None
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def spec_docs(folder: Path) -> list[CourseDoc]:
    """One document per `## ` section of spec.md (the preamble is its own section)."""
    text = (folder / "spec.md").read_text(encoding="utf-8")
    docs = []
    for i, section in enumerate(re.split(r"^## ", text, flags=re.MULTILINE)):
        title, _, body = section.strip().partition("\n")
        docs.append(CourseDoc(f"spec:{i}", "spec", title.lstrip("# "), body.strip() or title))
    return docs


def rubric_docs(folder: Path) -> list[CourseDoc]:
    assignment = load_assignment(folder)
    items = [*assignment.rubric, *assignment.policies]
    return [
        CourseDoc(
            f"rubric:{item.id}", "rubric", item.id, f"{item.description} ({item.marks} marks)"
        )
        for item in items
    ]


def announcement_docs(folder: Path) -> list[CourseDoc]:
    announcements = load_announcements(folder)
    superseded = {old: a.id for a in announcements for old in a.supersedes}
    return [
        CourseDoc(
            f"announcement:{a.id}",
            "announcement",
            a.title,
            a.body,
            date=a.date.isoformat(),
            superseded_by=superseded.get(a.id),
        )
        for a in announcements
    ]


class CourseDocIndex:
    def __init__(self, docs: list[CourseDoc]) -> None:
        self.docs = docs
        self._bm25 = BM25Okapi([tokenize(f"{d.title} {d.text}") for d in docs])

    @classmethod
    def for_assignment(cls, folder: Path) -> "CourseDocIndex":
        return cls([*spec_docs(folder), *rubric_docs(folder), *announcement_docs(folder)])

    def search(self, query: str, k: int = 3) -> list[CourseDoc]:
        """Top-k documents by BM25. An announcement that a later one supersedes always
        comes right after the announcement now in force, even if that one scored lower,
        so the rule in force is seen first."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: (-scores[i], i))
        hits = [self.docs[i] for i in ranked if scores[i] > 0]
        return in_force_first(hits, self.docs)[:k]


def in_force(doc: CourseDoc, by_id: dict[str, CourseDoc]) -> CourseDoc:
    """The announcement currently in force in doc's supersession chain."""
    seen = {doc.doc_id}
    while doc.superseded_by is not None:
        newer = by_id.get(f"announcement:{doc.superseded_by}")
        if newer is None or newer.doc_id in seen:
            break
        seen.add(newer.doc_id)
        doc = newer
    return doc


def in_force_first(hits: list[CourseDoc], all_docs: list[CourseDoc]) -> list[CourseDoc]:
    by_id = {d.doc_id: d for d in all_docs}
    ordered: list[CourseDoc] = []
    for doc in hits:
        current = in_force(doc, by_id)
        for item in (current, doc):
            if item not in ordered:
                ordered.append(item)
    return ordered
