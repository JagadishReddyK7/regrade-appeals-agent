"""The test-run configs must equal their frozen dev counterparts (docs/DECISIONS.md
section 15): only the name, the test split, the data root and the one setting an
ablation changes may differ."""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from eval.run import is_test_split, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
UNTUNED_3B = "Qwen/Qwen2.5-3B-Instruct"

# test config -> (dev config, the ablation flag it adds, or None)
PAIRS: dict[str, tuple[str, str | None]] = {
    "b0_reject_test": ("b0_reject_dev", None),
    "b0_accept_test": ("b0_accept_dev", None),
    "b1_test": ("b1_dev", None),
    "agent_test": ("agent_dev", None),
    "pipeline_test": ("pipeline_ft_dev", None),
    "pipeline_prompted_verifier_test": ("pipeline_dev", None),
    "pipeline_base_verifier_test": ("pipeline_ft_dev", None),
    "pipeline_no_claim_splitting_test": ("pipeline_ft_dev", "no_claim_splitting"),
    "pipeline_no_code_execution_test": ("pipeline_ft_dev", "no_code_execution"),
    "pipeline_no_retrieval_test": ("pipeline_ft_dev", "no_retrieval"),
    "pipeline_no_escalation_test": ("pipeline_ft_dev", "no_escalation"),
}


def read(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def expected(test_name: str) -> dict[str, Any]:
    dev_name, flag = PAIRS[test_name]
    config = copy.deepcopy(read(dev_name))
    config.update(name=test_name, cases="data/private/cases/test", data_root="data/private")
    if flag is not None:
        config["ablation"] = {flag: True}
    if test_name == "pipeline_base_verifier_test":
        config["verifier"]["llm"]["model"] = UNTUNED_3B
    return config


def test_every_test_config_is_listed() -> None:
    assert {p.stem for p in CONFIGS.glob("*_test.yaml")} == set(PAIRS)


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_test_config_matches_dev(name: str) -> None:
    config = read(name)
    assert config == expected(name)
    assert is_test_split(Path(config["cases"]))
    load_config(CONFIGS / f"{name}.yaml")


def test_main_system_uses_the_fine_tuned_verifier() -> None:
    verifier = read("pipeline_test")["verifier"]["llm"]
    assert verifier["model"] == "/kaggle/temp/verifier-ft"
    assert verifier["serve"]["gpu"] == 1
    assert read("pipeline_prompted_verifier_test")["verifier"]["llm"] is None
