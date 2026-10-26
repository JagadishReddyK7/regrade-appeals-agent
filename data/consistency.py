"""Checks across cases: tone variants of one grading situation, equivalence groups (RQ2)
and minimal pairs (RQ2) must agree on everything except how the appeal is written.

`base_of` maps a case to its original submission, so a hidden-instruction variant (a copy
of the submission with an added comment) counts as the same submission.
"""

from collections import defaultdict
from collections.abc import Callable, Hashable
from pathlib import Path

from agent.schemas import Case

BaseOf = Callable[[Case], str]


def claims_key(case: Case) -> Hashable:
    return tuple(sorted((c.target, c.tier, c.disputed_marks, c.gold_verdict) for c in case.claims))


def planted_key(case: Case) -> Hashable:
    error = case.planted_error
    return None if error is None else (error.type, error.target, error.marks)


def answers_key(case: Case) -> Hashable:
    return (
        case.answer,
        case.followup.has_new_evidence,
        case.followup.answer_after,
    )


def situation_key(case: Case, base_of: BaseOf) -> Hashable:
    """Tone variants share the assignment, submission, error, claims and answers."""
    return (case.assignment, base_of(case), planted_key(case), claims_key(case), answers_key(case))


def group_key(case: Case, base_of: BaseOf) -> Hashable:
    """Equivalence-group members share the planted error and the correct answers."""
    return (case.assignment, planted_key(case), answers_key(case))


def pair_key(case: Case, base_of: BaseOf) -> Hashable:
    """Minimal pairs share the submission, claims and answers; only the wording differs."""
    return (case.assignment, base_of(case), planted_key(case), claims_key(case), answers_key(case))


def grouped(cases: dict[Path, Case], attr: str) -> dict[str, list[tuple[Path, Case]]]:
    groups: dict[str, list[tuple[Path, Case]]] = defaultdict(list)
    for path, case in cases.items():
        value = getattr(case, attr)
        if value is not None:
            groups[value].append((path, case))
    return groups


def mismatches(
    members: list[tuple[Path, Case]], key: Callable[[Case], Hashable], label: str
) -> dict[Path, list[str]]:
    """Members whose key differs from the first member's."""
    first_path, first = members[0]
    expected = key(first)
    return {
        path: [f"{label}: differs from {first.case_id} ({first_path.name})"]
        for path, case in members[1:]
        if key(case) != expected
    }


def check_consistency(cases: dict[Path, Case], base_of: BaseOf) -> dict[Path, list[str]]:
    errors: dict[Path, list[str]] = defaultdict(list)

    def add(found: dict[Path, list[str]]) -> None:
        for path, messages in found.items():
            errors[path] += messages

    for sid, members in grouped(cases, "situation_id").items():
        add(mismatches(members, lambda c: situation_key(c, base_of), f"situation {sid}"))
        tones = [case.tone for _, case in members]
        for path, case in members:
            if tones.count(case.tone) > 1:
                errors[path].append(f"situation {sid}: tone {case.tone!r} appears more than once")
    for gid, members in grouped(cases, "group_id").items():
        add(mismatches(members, lambda c: group_key(c, base_of), f"equivalence group {gid}"))
        bases = [base_of(case) for _, case in members]
        for path, case in members:
            if bases.count(base_of(case)) > 1:
                errors[path].append(f"equivalence group {gid}: submissions must all differ")
    for pid, members in grouped(cases, "pair_id").items():
        add(mismatches(members, lambda c: pair_key(c, base_of), f"minimal pair {pid}"))
        if len(members) != 2:
            for path, _ in members:
                errors[path].append(f"minimal pair {pid} has {len(members)} cases, not 2")
    return dict(errors)
