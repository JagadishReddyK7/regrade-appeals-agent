import importlib

import pytest

PACKAGES = ["sandbox", "llm", "agent", "retrieval", "verifier", "baselines", "eval"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    assert importlib.import_module(name) is not None
