## What and why
<!-- One or two sentences. Link the issue: "Closes #12". -->

## How it was tested
<!-- Tests added or changed; any manual run (dev cases, Kaggle). -->

## Checklist (author fills in, reviewer checks)
- [ ] Tests added or updated; `python -m pytest` and `python -m ruff check .` pass locally
- [ ] `python -m data.validate data/cases/dev` passes (if data, schemas or the grader changed)
- [ ] No test-split data: nothing from `data/cases/test/`, `data/private/` or blind cases
- [ ] No answer key reaches a system: tools and prompts never read `reference_grade.json`, the reference solution or gold case fields
- [ ] No secrets, `.env`, model weights, caches, `runs/` or `.claude/settings.local.json`
- [ ] Every commit passes on its own (all files its code imports are in it)
- [ ] Metric definitions unchanged, or the change was agreed and is recorded
- [ ] `docs/DECISIONS.md` updated for any design decision or deviation from the proposal
- [ ] `docs/STATUS.md` updated if this finishes a milestone-level item; the issue is updated
