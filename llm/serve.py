"""Start a vLLM OpenAI-compatible server from a config's `llm` section and wait until
it answers.

Usage: python -m llm.serve --config configs/b1_dev.yaml [--verifier]
Starts the server in the background (its output goes to vllm.log), waits until
/v1/models responds and exits; the server keeps running. Stop it with `kill <pid>`;
the pid is written to vllm.pid. With --verifier it starts the server of the config's
`verifier.llm` section instead (e.g. the fine-tuned 3B on the second GPU), logging to
vllm_verifier.log and vllm_verifier.pid. `serve.base_model` names the model to load when
it differs from the name requests use (`model`).
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

Fetch = Callable[[str], bool]


def build_command(llm: dict[str, Any]) -> list[str]:
    serve = llm.get("serve", {})
    cmd = [
        "vllm",
        "serve",
        serve.get("base_model", llm["model"]),
        "--port",
        str(serve.get("port", 8000)),
        "--dtype",
        serve.get("dtype", "half"),
        "--max-model-len",
        str(serve.get("max_model_len", 8192)),
        "--gpu-memory-utilization",
        str(serve.get("gpu_memory_utilization", 0.85)),
        "--seed",
        "0",
    ]
    if serve.get("quantization"):
        cmd += ["--quantization", serve["quantization"]]
    if serve.get("tool_call_parser"):
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", serve["tool_call_parser"]]
    return cmd


def server_section(config: dict[str, Any], verifier: bool) -> dict[str, Any]:
    """The `llm` section, or with `verifier` the `verifier.llm` section."""
    if not verifier:
        return config["llm"]
    llm = (config.get("verifier") or {}).get("llm")
    if not llm:
        raise ValueError("the config has no verifier.llm section (the verifier is the main LLM)")
    return llm


def server_env(llm: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(llm.get("serve", {}).get("gpu", 0))
    return env


def models_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def wait_until_ready(
    base_url: str,
    timeout_s: float = 900,
    poll_s: float = 5,
    fetch: Fetch = models_ready,
    is_alive: Callable[[], bool] = lambda: True,
    log_name: str = "vllm.log",
) -> None:
    """Poll {base_url}/models until it answers. Raises if the server dies or times out."""
    url = base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if fetch(url):
            return
        if not is_alive():
            raise RuntimeError(f"vLLM server exited before it was ready; see {log_name}")
        time.sleep(poll_s)
    raise TimeoutError(f"vLLM server not ready after {timeout_s} s; see {log_name}")


def start_server(llm: dict[str, Any], log_path: Path = Path("vllm.log")) -> subprocess.Popen[bytes]:
    """Start vLLM detached from this process and wait until it answers."""
    log = open(log_path, "wb")  # noqa: SIM115  # stays open for the server's lifetime
    proc = subprocess.Popen(
        build_command(llm),
        env=server_env(llm),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    wait_until_ready(llm["base_url"], is_alive=lambda: proc.poll() is None, log_name=log_path.name)
    return proc


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm.serve")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verifier", action="store_true", help="serve verifier.llm")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    llm = server_section(config, args.verifier)
    stem = "vllm_verifier" if args.verifier else "vllm"
    print("starting:", " ".join(build_command(llm)), flush=True)
    proc = start_server(llm, Path(f"{stem}.log"))
    Path(f"{stem}.pid").write_text(str(proc.pid), encoding="utf-8")
    print(f"vLLM ready at {llm['base_url']} (pid {proc.pid})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
