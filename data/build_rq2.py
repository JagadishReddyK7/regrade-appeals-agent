"""Build the RQ2 consistency cases (equivalence groups and minimal pairs) from set files.

Usage: python -m data.build_rq2 <set.yaml>... [--data-root data] [--out data/cases/rq2]

A *set file* describes one equivalence group or one minimal pair. Everything the members
share (assignment, planted error, claims, answers) is written once; each member is one case
with its own submission (groups) or the same submission (pairs), its own tone and wording.
Unlike `data/build_cases.py`, which writes the four tone variants of one submission, every
member here is a single case, so each gets its own `situation_id` (`<set_id>-<n>`); the set
id becomes the case's `group_id` or `pair_id`. The validator then checks the RQ2 rules
(group members differ in submission but share error and answers; a pair has exactly two
cases with the same submission and content). See docs/DECISIONS.md section 16.
"""

import argparse
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from agent.schemas import (
    Answer,
    Case,
    Claim,
    FollowUp,
    NonEmptyStr,
    PlantedError,
    Source,
    Split,
    StrictModel,
    Tone,
)
from data.build_cases import BuildError, SituationClaim, SituationFollowUp, case_json
from data.validate import validate_folder


class Member(StrictModel):
    case_id: str = NonEmptyStr
    submission: str = NonEmptyStr
    tone: Tone
    wording: str = NonEmptyStr  # e.g. fluent, less_fluent, assertive, hesitant (for readers)
    appeal_text: str = NonEmptyStr
    spans: dict[str, str]  # claim id -> verbatim span of appeal_text
    followup_text: str = NonEmptyStr


class ConsistencySet(StrictModel):
    set_id: str = NonEmptyStr
    kind: Literal["group", "pair"]
    assignment: str = NonEmptyStr
    split: Split
    source: Source
    planted_error: PlantedError | None
    claims: list[SituationClaim]
    answer: Answer
    followup: SituationFollowUp
    members: list[Member] = Field(min_length=2)


def load_set(path: Path) -> ConsistencySet:
    try:
        return ConsistencySet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise BuildError(f"{path}: {details}") from exc


def build_set(cset: ConsistencySet) -> list[Case]:
    claim_ids = {c.id for c in cset.claims}
    cases = []
    for number, member in enumerate(cset.members, start=1):
        if set(member.spans) != claim_ids:
            raise BuildError(f"{cset.set_id} member {number}: spans must cover {sorted(claim_ids)}")
        try:
            claims = [Claim(span=member.spans[c.id], **c.model_dump()) for c in cset.claims]
            cases.append(
                Case(
                    case_id=member.case_id,
                    situation_id=f"{cset.set_id}-{number}",
                    assignment=cset.assignment,
                    submission=member.submission,
                    tone=member.tone,
                    source=cset.source,
                    appeal_text=member.appeal_text,
                    planted_error=cset.planted_error,
                    claims=claims,
                    followup=FollowUp(
                        text=member.followup_text,
                        has_new_evidence=cset.followup.has_new_evidence,
                        answer_after=cset.followup.answer_after,
                    ),
                    answer=cset.answer,
                    group_id=cset.set_id if cset.kind == "group" else None,
                    pair_id=cset.set_id if cset.kind == "pair" else None,
                    split=cset.split,
                )
            )
        except ValidationError as exc:
            details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise BuildError(f"{cset.set_id} member {number}: {details}") from exc
    return cases


def write_cases(cases: list[Case], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for case in cases:
        path = out_dir / f"{case.case_id}.json"
        path.write_text(case_json(case), encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m data.build_rq2")
    parser.add_argument("sets", type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/cases/rq2"))
    args = parser.parse_args(argv)
    try:
        for path in args.sets:
            cset = load_set(path)
            written = write_cases(build_set(cset), args.out)
            print(f"{cset.set_id}: {', '.join(p.name for p in written)}")
    except BuildError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    status = 0
    for path, errors in validate_folder(args.out, args.data_root).items():
        for error in errors:
            print(f"INVALID {path}: {error}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
