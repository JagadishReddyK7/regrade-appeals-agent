# Code architecture

How the code fits together and how to extend it. Why it looks this way is in
`docs/DECISIONS.md`; the data formats are in `docs/DATA_FORMAT.md`.

## How one evaluation run flows

```
python -m eval.run --config configs/<name>.yaml
  eval/run.py  run_eval(config)
    load_cases(config.cases)                 agent/schemas.py  Case (refuses cases/test)
    RunLogger(runs/<run id>/events.jsonl)    eval/runlog.py    one JSON line per event
    backend_from_config(config.llm)          llm/client.py     OpenAIBackend (vLLM) | MockLLM in tests
    build_system(config, ...)                -> a System (baselines/base.py)
    for each case:
      run_case(system, case, logger)         a crash = case_error, counted as no decision
        system.run(case, logger) -> SystemOutput(first, second, followup_run)
      CaseRecord.from_case(case, output)     eval/metrics.py   joins prediction and answer key
    compute_all(records) -> metrics.json
```

### Inside the systems

**B1** (`baselines/b1_direct.py`): `course_context()` builds one prompt from the
assignment's spec, rubric, announcements, the grade report and the submission (never the
answer key). `LLMClient.chat(..., json_schema=DirectDecisionOutput)` -> validated ->
`Decision`. With the follow-up enabled, the follow-up text is appended and it decides again.

**Agent v0** (`AgentSystem` in `eval/run.py` -> `agent/loop.py`; options and ablation
flags come from the config through `AgentOptions.from_config`):
```
AgentLoop.run()
  messages = system prompt + appeal (assignment and submission ids only;
             + the rubric when retrieval is ablated)
  turn 1: repeat until submit_decision or max_tool_calls (10):
    LLMClient.chat(messages, tools=tool_schemas(minus ablated tools))
    for each tool call:
      validate arguments            llm/parsing.py + agent/schemas.py (*Args models)
      ToolBox.call(name, args)      agent/tools.py
        run_tests / run_code   -> sandbox/runner.py SandboxRunner (subprocess + cache)
        search_course_docs     -> retrieval/course_docs.py (BM25, in-force first)
        search_precedents      -> retrieval/precedents.py (configured store or empty)
        get_grade_report etc.  -> data/formats.py loaders
        propose_grade_change   -> proposals.jsonl (capped by deducted marks)
      result gets an evidence id E<n>; shortened for the model, full in the log
    submit_decision -> checked -> Decision (claim id "appeal")
  turn 2 (if followup and turn 1 decided): + follow-up text, fresh budget, same loop
  -> AgentResult(decision, followup_decision | None, counters)
```

### What each part may see

| Part | Sees | Never sees |
| --- | --- | --- |
| Systems (agent, baselines) | appeal text, assignment/submission ids, tools or documents | `reference_grade.json`, reference solution, gold case fields |
| Tools (`agent/tools.py`) | submission, grade report, the autograder that graded it, course docs | answer key |
| Metrics (`eval/metrics.py`) | predictions and the answer key | - |
| Validator / reference grader (`data/`) | everything, including the answer key | - |

## Module map

| Module | Main contents |
| --- | --- |
| `agent/schemas.py` | `Case`, `Claim`, `Answer`, `FollowUp`, `PlantedError`; `Decision`, `ClaimVerdict`; tool argument models; `DirectDecisionOutput` |
| `agent/tools.py` | `TOOL_SPECS` (name, description, argument model), `ToolBox` (implementations), `ToolError`, `RejectedAction` |
| `agent/loop.py` | `AgentLoop`, `run_agent`, `AgentResult`, output shortening, system prompt |
| `baselines/base.py` | `System` protocol, `SystemOutput` |
| `baselines/b1_direct.py` | `DirectBaseline` |
| `data/formats.py` | `Assignment`, `RubricItem`, `Policy`, `Announcement`, `GradeReport`, `ReferenceGrade`, loaders, `TOTAL_ID` |
| `data/grader.py` | reference grader (`reference_total`), `deducted_marks`, per-case and per-assignment checks |
| `data/consistency.py` | checks across cases (situations, equivalence groups, minimal pairs) |
| `data/validate.py` | `python -m data.validate` |
| `data/build_cases.py` | `Situation`, `build_situation`, `python -m data.build_cases`: one situation file -> four tone variants (+ hidden-instruction submission copy), then validates |
| `eval/metrics.py` | `Prediction`, `CaseRecord`, the section 11 metrics, `METRICS`, `compute_all`, `by_group` |
| `eval/run.py` | config loading, `build_system`, `AgentSystem`, `run_eval`, `code_version` |
| `eval/runlog.py` | `RunLogger`, `RunContext`, `config_hash`, `read_events` |
| `eval/cost.py` | cost per appeal (wall time, LLM calls, latency, tokens, tool calls) |
| `llm/client.py` | `LLMClient`, `OpenAIBackend`, `MockLLM`, `ChatResponse`, `ToolCall` |
| `llm/parsing.py` | strict JSON parsing; `validate_output` logs `invalid_output` |
| `llm/serve.py`, `llm/smoke.py` | start vLLM from a config; check chat, tool calls, constrained JSON |
| `retrieval/course_docs.py` | `CourseDocIndex` (BM25 over spec sections, rubric items, announcements; the announcement in force first) |
| `retrieval/precedents.py` | `Precedent`, `PrecedentStore` (JSONL, BM25 per assignment) |
| `verifier/format.py` | verifier prompt and answer format, evidence bundles |
| `verifier/train.py`, `verifier/dummy_data.py` | QLoRA training (GPU), dummy examples |
| `sandbox/runner.py`, `sandbox/harness.py` | `SandboxRunner.run_test` / `run_code`; the child process that applies limits and blocks |
| `sandbox/cache.py`, `sandbox/result.py` | result cache; `RunResult` |

## Events in `events.jsonl`

Every record has `ts`, `event`, `run_id`, `case_id`, `seed`, `config_hash`.

| Event | Written by | Main fields |
| --- | --- | --- |
| `run_start` | `eval/run.py` | `code_version` (commit, dirty), `config` |
| `llm_call` / `llm_error` | `LLMClient.chat` | model, prompt_hash, tokens, latency_s, content, tool_calls / error |
| `invalid_output` | `llm/parsing.py`, B1 | kind, schema, error, raw |
| `tool_call` | `AgentLoop` | tool, arguments, status (ok/invalid/error/rejected), output, error, truncated_for_model |
| `decision` | `AgentLoop`, B1 | decision, status/turn, counters |
| `case_error`, `followup_error` | `eval/run.py`, B1 | error, traceback |

`proposals.jsonl` holds `grade_change_proposal` and `escalation` records (status `pending_ta`).

## How to extend

**Add a system (baseline or agent variant)**
1. Write a class with `run(self, case, logger) -> SystemOutput` (see `baselines/base.py`).
   Log through the given `logger`; build an `LLMClient(backend, logger)` for model calls.
2. Register it: add its name to `SYSTEMS` and a branch to `build_system` in `eval/run.py`.
3. Add `configs/<name>_dev.yaml` and tests with `MockLLM` (see `tests/test_eval_run.py`).

**Add a tool**
1. Add an argument model to `agent/schemas.py` (subclass `StrictModel`).
2. Add a `ToolSpec` to `TOOL_SPECS` and a handler method to `ToolBox` (register it in
   `self._handlers`). Raise `ToolError` for failures the model should see and
   `RejectedAction` for safety refusals. Never read answer-key files.
3. Update the tool list in `CLAUDE.md` and the test that lists tool names.

**Add a metric**
Only with team agreement (`CLAUDE.md`: metric definitions do not change without explicit
instruction). Write a function `(records) -> float | None` that returns `None` for an
empty denominator, add it to `METRICS`, record its definition in `docs/DECISIONS.md`
section 7, and test it on hand-made `CaseRecord`s (see `tests/test_metrics.py`).

**Add a config option or ablation flag**
Ablations A0-A7 are config flags, not code forks. Read the option from the config dict
where the system is built (`build_system` / the system's constructor) with a default that
keeps current behaviour, and pass it down; do not read config deep inside helpers. The
config hash in every event changes automatically, so runs stay distinguishable. Agent
ablations live in the `ablation:` section and are parsed by `AgentOptions.from_config` in
`agent/loop.py`: add the flag to `ABLATION_FLAGS` and `AgentOptions`, add a
`configs/agent_<flag>_dev.yaml`, and record it in `docs/DECISIONS.md` section 10.

**Add benchmark data**
See `docs/DATA_FORMAT.md`; run `python -m data.validate` before committing.

## Testing patterns

- **Model calls:** script responses with `MockLLM([...])`; it records each request in
  `.requests`, so tests can assert on prompts and tool schemas.
- **Data:** tests that modify data copy `data/assignments` and `data/submissions` into
  `tmp_path` first (see `data_root` fixture in `tests/test_grader.py`).
- **Sandbox:** share one `SandboxRunner(cache=ResultCache(tmp))` per test module to keep
  the suite fast; Linux-only limit tests use the `posix_only` marker.
- **Logs:** write to `tmp_path` and read back with `read_events(path, "event_name")`.
