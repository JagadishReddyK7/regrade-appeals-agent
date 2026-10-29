import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.schemas import EvidenceBundle, EvidenceItem, VerifierExample, VerifierOutput
from verifier.dummy_data import TRAINING_TONES, dummy_examples, read_examples, write_examples
from verifier.format import (
    answer_text,
    bundle_for_claim,
    parse_answer,
    prompt_messages,
    training_messages,
)
from verifier.train import (
    IGNORE_INDEX,
    Encoded,
    TrainConfig,
    batches,
    encode,
    load_train_config,
    pad_batch,
    tokenize_example,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = EvidenceBundle(
    claim="My function returns 0 for an empty list, sir, please!",
    target="test_empty",
    evidence=[EvidenceItem(id="E1", source="run_code", text='{"stdout": "0"}')],
)


# --- Schemas ----------------------------------------------------------------------------


def test_decided_verdict_must_cite_evidence() -> None:
    with pytest.raises(ValidationError, match="must cite"):
        VerifierOutput(verdict="supported")
    assert VerifierOutput(verdict="needs_human").evidence_id is None


def test_example_label_must_cite_bundle_evidence() -> None:
    with pytest.raises(ValidationError, match="unknown evidence"):
        VerifierExample(
            example_id="x",
            bundle=BUNDLE,
            label=VerifierOutput(verdict="supported", evidence_id="E9"),
            tone="polite",
            assignment="D1",
        )


def test_bundle_rejects_duplicate_evidence_ids() -> None:
    item = EvidenceItem(id="E1", source="run_code", text="x")
    with pytest.raises(ValidationError, match="duplicate"):
        EvidenceBundle(claim="c", evidence=[item, item])


# --- Prompt format ----------------------------------------------------------------------


def test_bundle_from_agent_evidence_is_shortened() -> None:
    evidence = {
        "E1": {"tool": "run_tests", "output": {"stderr": "x" * 10_000}},
        "E2": {"tool": "get_submission", "output": {"code": "def f(): pass"}},
    }
    bundle = bundle_for_claim("claim as written", "test_a", evidence)
    assert [(i.id, i.source) for i in bundle.evidence] == [
        ("E1", "run_tests"),
        ("E2", "get_submission"),
    ]
    assert len(bundle.evidence[0].text) <= 1500
    assert "def f(): pass" in bundle.evidence[1].text


def test_prompt_keeps_claim_verbatim_and_lists_evidence() -> None:
    system, user = prompt_messages(BUNDLE)
    assert system["role"] == "system" and "not evidence" in system["content"]
    assert BUNDLE.claim in user["content"]  # tone included: the verifier sees the claim as written
    assert "[E1] from run_code" in user["content"] and "test_empty" in user["content"]


def test_training_messages_end_with_answer() -> None:
    example = dummy_examples(1)[0]
    messages = training_messages(example)
    assert messages[-1] == {"role": "assistant", "content": answer_text(example.label)}
    assert json.loads(messages[-1]["content"])["verdict"] == example.label.verdict


@pytest.mark.parametrize(
    ("raw", "ok"),
    [
        ('{"verdict": "supported", "evidence_id": "E1"}', True),
        ('{"verdict": "needs_human", "evidence_id": null}', True),
        ('{"verdict": "supported", "evidence_id": "E7"}', False),  # not in the bundle
        ('{"verdict": "supported", "evidence_id": null}', False),  # decided without evidence
        ('```json\n{"verdict": "refuted", "evidence_id": "E1"}\n```', False),  # no repair
    ],
)
def test_parse_answer(raw: str, ok: bool) -> None:
    assert parse_answer(raw, BUNDLE).ok is ok


# --- Dummy data -------------------------------------------------------------------------


def test_dummy_examples_are_tone_paired() -> None:
    examples = dummy_examples(50, seed=0)
    assert len(examples) == 50
    assert {e.tone for e in examples} <= set(TRAINING_TONES)
    assert "authority" not in {e.tone for e in examples}  # held out, as planned
    labels: dict[str, set[str]] = defaultdict(set)
    claims: dict[str, set[str]] = defaultdict(set)
    for e in examples:
        labels[str(e.pair_id)].add(e.label.verdict)
        claims[str(e.pair_id)].add(e.bundle.claim)
    assert all(len(v) == 1 for v in labels.values())  # same label in every tone
    assert any(len(v) > 1 for v in claims.values())  # different wording
    assert set(Counter(e.label.verdict for e in examples)) == {
        "supported",
        "refuted",
        "needs_human",
    }


def test_dummy_examples_are_deterministic(tmp_path: Path) -> None:
    assert dummy_examples(20, seed=1) == dummy_examples(20, seed=1)
    path = tmp_path / "d.jsonl"
    write_examples(dummy_examples(10), path)
    assert read_examples(path) == dummy_examples(10)


# --- Training helpers (no GPU) ----------------------------------------------------------


def test_smoke_config_loads() -> None:
    config = load_train_config(ROOT / "configs" / "verifier_smoke.yaml")
    assert config.model == "Qwen/Qwen2.5-3B-Instruct"
    assert config.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"name": "x", "model": "m", "data": "d", "max_steps": 0})
    with pytest.raises(TypeError):
        TrainConfig.from_dict({"name": "x", "model": "m", "data": "d", "epochs": 3})


def test_encode_masks_prompt() -> None:
    encoded = encode([1, 2, 3], [7, 8], max_length=10)
    assert encoded == Encoded(input_ids=[1, 2, 3, 7, 8], labels=[IGNORE_INDEX] * 3 + [7, 8])
    assert encode([1] * 9, [7, 8], max_length=10) is None  # never truncate the prompt
    assert encode([1], [], max_length=10) is None


def test_pad_batch() -> None:
    batch = pad_batch([Encoded([1, 2, 3], [-100, 2, 3]), Encoded([4], [4])], pad_id=0)
    assert batch["input_ids"] == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"] == [[-100, 2, 3], [4, -100, -100]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


def test_batches_cycle_through_epochs() -> None:
    data = [Encoded([i], [i]) for i in range(5)]
    out = batches(data, batch_size=2, steps=7, seed=0)
    assert len(out) == 7
    first_epoch = [e for b in out[:3] for e in b]
    assert sorted(e.input_ids[0] for e in first_epoch) == [0, 1, 2, 3, 4]


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):  # noqa: ANN001
        assert not tokenize and add_generation_prompt
        return "".join(m["content"] for m in messages) + "<assistant>"

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": [len(word) for word in text.split()]}


def test_tokenize_example_appends_eos_to_answer() -> None:
    example = dummy_examples(1)[0]
    prompt_ids, answer_ids = tokenize_example(FakeTokenizer(), example)
    assert prompt_ids and answer_ids
    expected = answer_text(example.label) + "<eos>"
    assert answer_ids == FakeTokenizer()(expected, add_special_tokens=False)["input_ids"]
