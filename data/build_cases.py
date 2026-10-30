"""Build the four tone variants of a grading situation from one YAML file.

Usage: python -m data.build_cases <situation.yaml>... [--data-root data] [--out <folder>]

A situation file holds everything the tone variants share (assignment, submission, planted
error, claims, answers) and, per tone, the appeal text, the verbatim span of each claim and
the follow-up text. The builder creates the four case JSON files through the `Case` model
(so every schema rule applies), creates the hidden-instruction copy of the submission if it
does not exist, and then runs the validator on the output folder. The output folder
defaults to `<data-root>/cases/<split>`. Case ids are taken from `case_ids`, else from an
existing case of the same situation and tone (so rebuilding is stable), else the next free
`<split>-NNN`.

Private data: use `--data-root data/private` and keep the result out of Git
(see docs/DATA_FORMAT.md).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError, model_validator

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
    Tier,
    Tone,
    Verdict,
)
from data.formats import load_assignment
from data.validate import validate_folder

TONES: tuple[Tone, ...] = ("polite", "emotional", "authority", "hidden_instruction")


class BuildError(ValueError):
    """A situation file or the data it refers to cannot be built into valid cases."""


class SituationClaim(StrictModel):
    id: str = NonEmptyStr
    tier: Tier
    target: str = NonEmptyStr
    disputed_marks: float
    gold_verdict: Verdict


class SituationFollowUp(StrictModel):
    has_new_evidence: bool
    answer_after: Answer


class HiddenInstruction(StrictModel):
    comment: str = NonEmptyStr  # comment lines added to the submission's code
    submission: str | None = None  # name of the copy; default "<submission>-hi"


class ToneText(StrictModel):
    appeal_text: str = NonEmptyStr
    spans: dict[str, str]  # claim id -> verbatim span of appeal_text
    followup_text: str = NonEmptyStr


class Situation(StrictModel):
    situation_id: str = NonEmptyStr
    assignment: str = NonEmptyStr
    submission: str = NonEmptyStr
    split: Split
    source: Source
    planted_error: PlantedError | None
    claims: list[SituationClaim]
    answer: Answer
    followup: SituationFollowUp
    hidden_instruction: HiddenInstruction
    tones: dict[Tone, ToneText]
    case_ids: dict[Tone, str] = {}

    @model_validator(mode="after")
    def _all_tones_with_spans(self) -> "Situation":
        missing = [t for t in TONES if t not in self.tones]
        if missing:
            raise ValueError(f"missing tones: {missing}")
        claim_ids = {c.id for c in self.claims}
        for tone, text in self.tones.items():
            if set(text.spans) != claim_ids:
                raise ValueError(
                    f"tone {tone}: spans must be given for exactly {sorted(claim_ids)}"
                )
        return self

    @property
    def hidden_submission(self) -> str:
        return self.hidden_instruction.submission or f"{self.submission}-hi"


def load_situation(path: Path) -> Situation:
    try:
        return Situation.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise BuildError(f"{path}: {details}") from exc


# --- Hidden-instruction submission ------------------------------------------------------


def ensure_hidden_variant(data_root: Path, situation: Situation) -> Path | None:
    """Create the hidden-instruction copy of the submission unless it already exists.
    Returns the new folder, or None if the copy existed (an existing copy is never changed;
    the validator checks it differs from the original only by comments)."""
    submissions = data_root / "submissions"
    base, variant = submissions / situation.submission, submissions / situation.hidden_submission
    if variant.exists():
        return None
    if not base.is_dir():
        raise BuildError(f"submission {situation.submission} not found in {submissions}")
    comment = situation.hidden_instruction.comment.strip("\n")
    if any(line.strip() and not line.lstrip().startswith("#") for line in comment.splitlines()):
        raise BuildError("hidden_instruction.comment must consist of comment lines only")
    module = load_assignment(data_root / "assignments" / situation.assignment).module
    shutil.copytree(base, variant)
    code = variant / module
    code.write_text(f"{comment}\n\n\n{code.read_text(encoding='utf-8')}", encoding="utf-8")
    report_path = variant / "grade_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["submission"] = situation.hidden_submission
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ref_path = variant / "reference_grade.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref["variant_of"] = situation.submission
    ref_path.write_text(json.dumps(ref) + "\n", encoding="utf-8")
    return variant


# --- Cases ------------------------------------------------------------------------------


def existing_ids(out_dir: Path) -> tuple[dict[tuple[str, str], str], set[str]]:
    """Case ids by (situation, tone) of the cases already in out_dir, and all ids in use."""
    by_situation: dict[tuple[str, str], str] = {}
    used: set[str] = set()
    for path in sorted(out_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        by_situation[(data["situation_id"], data["tone"])] = data["case_id"]
        used.add(data["case_id"])
    return by_situation, used


def assign_case_ids(situation: Situation, out_dir: Path) -> dict[Tone, str]:
    """Explicit ids first, then the ids of existing cases of this situation, then new ones."""
    by_situation, used = existing_ids(out_dir)
    ids: dict[Tone, str] = {}
    for tone in TONES:
        known = situation.case_ids.get(tone) or by_situation.get((situation.situation_id, tone))
        if known is not None:
            ids[tone] = known
    for tone, case_id in ids.items():
        owner = next((key for key, cid in by_situation.items() if cid == case_id), None)
        if owner is not None and owner != (situation.situation_id, tone):
            raise BuildError(f"case id {case_id} is already used by {owner}")
    taken = used | set(ids.values())
    number = 1
    for tone in TONES:
        if tone in ids:
            continue
        while f"{situation.split}-{number:03d}" in taken:
            number += 1
        ids[tone] = f"{situation.split}-{number:03d}"
        taken.add(ids[tone])
    return ids


def build_cases(situation: Situation, case_ids: dict[Tone, str]) -> list[Case]:
    cases = []
    for tone in TONES:
        text = situation.tones[tone]
        try:
            claims = [Claim(span=text.spans[c.id], **c.model_dump()) for c in situation.claims]
            cases.append(
                Case(
                    case_id=case_ids[tone],
                    situation_id=situation.situation_id,
                    assignment=situation.assignment,
                    submission=(
                        situation.hidden_submission
                        if tone == "hidden_instruction"
                        else situation.submission
                    ),
                    tone=tone,
                    source=situation.source,
                    appeal_text=text.appeal_text,
                    planted_error=situation.planted_error,
                    claims=claims,
                    followup=FollowUp(
                        text=text.followup_text,
                        has_new_evidence=situation.followup.has_new_evidence,
                        answer_after=situation.followup.answer_after,
                    ),
                    answer=situation.answer,
                    split=situation.split,
                )
            )
        except ValidationError as exc:
            details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise BuildError(f"{situation.situation_id} tone {tone}: {details}") from exc
    return cases


def whole_numbers(value: Any) -> Any:
    """Marks are floats in the model; write 5.0 as 5, like the hand-written case files."""
    if isinstance(value, dict):
        return {k: whole_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [whole_numbers(v) for v in value]
    return int(value) if isinstance(value, float) and value.is_integer() else value


def case_json(case: Case) -> str:
    data = whole_numbers(case.model_dump(mode="json"))
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def build_situation(situation: Situation, data_root: Path, out_dir: Path) -> list[Path]:
    """Write the situation's four case files into out_dir; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(situation, assign_case_ids(situation, out_dir))
    ensure_hidden_variant(data_root, situation)
    paths = []
    for case in cases:
        path = out_dir / f"{case.case_id}.json"
        path.write_text(case_json(case), encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m data.build_cases")
    parser.add_argument("situations", type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path, help="output folder (default: <data-root>/cases/<split>)"
    )
    args = parser.parse_args(argv)
    folders: set[Path] = set()
    try:
        for path in args.situations:
            situation = load_situation(path)
            out_dir = args.out or args.data_root / "cases" / situation.split
            written = build_situation(situation, args.data_root, out_dir)
            folders.add(out_dir)
            print(f"{situation.situation_id}: {', '.join(p.name for p in written)}")
    except BuildError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    status = 0
    for folder in sorted(folders):
        for path, errors in validate_folder(folder, args.data_root).items():
            for error in errors:
                print(f"INVALID {path}: {error}", file=sys.stderr)
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
