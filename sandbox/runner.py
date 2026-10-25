"""Run untrusted student code in a resource-limited subprocess.

Isolation provided:
- a separate Python process started with a clean environment, a fresh temporary working
  directory, and the repository kept off sys.path;
- POSIX only (Linux on Kaggle): CPU-time, address-space and file-size limits via the
  `resource` module, applied in the child before student code is imported;
- everywhere: a wall-clock timeout that kills the whole process group;
- networking modules (`socket`, `ssl` and their C cores) cannot be imported.

Limitations: this guards against accidental damage, not a deliberately adversarial
program. Without containers or OS-level sandboxing the network cannot be fully blocked
(e.g. ctypes can call the C socket API directly), the file system outside the working
directory is still reachable, child processes are not limited, and student code runs in
the same process as the harness, so it could tamper with its own result. On Windows,
CPU and memory limits are not enforced.
"""

import ast
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sandbox.cache import CacheKey, ResultCache, sha256_bytes
from sandbox.result import ErrorType, RunResult

REPORTED_STATUSES = {"test_failed", "exception", "memory_limit", "network_blocked"}

HARNESS = Path(__file__).with_name("harness.py")
HARNESS_HASH = sha256_bytes(HARNESS.read_bytes())  # cached results expire when it changes
IS_POSIX = os.name == "posix"
CUSTOM_TEST_ID = "<custom>"
TRUNCATION_NOTE = "\n[output truncated]"


@dataclass(frozen=True)
class Limits:
    cpu_seconds: int = 5
    memory_mb: int = 512
    wall_seconds: float = 10.0
    max_output_bytes: int = 64 * 1024

    def child_config(self) -> dict[str, int]:
        fsize = max(4 * self.max_output_bytes, 1024 * 1024)
        return {"cpu_seconds": self.cpu_seconds, "memory_mb": self.memory_mb, "fsize_bytes": fsize}


@dataclass(frozen=True)
class Outcome:
    returncode: int | None
    timed_out: bool
    report: dict[str, Any] | None
    stdout: str
    stderr: str
    runtime_s: float


class SandboxRunner:
    def __init__(
        self,
        cache: ResultCache | None = None,
        limits: Limits | None = None,
        python: str = sys.executable,
    ) -> None:
        self.cache = cache
        self.limits = limits or Limits()
        self.python = python

    def run_test(self, submission: Path, tests: Path, test_id: str) -> RunResult:
        """Run one named test function from `tests` against `submission`."""
        tests_source = tests.read_bytes()
        if test_id not in function_names(tests_source):
            raise ValueError(f"unknown test {test_id!r} in {tests.name}")
        if submission.name == tests.name:
            raise ValueError("submission and tests file must have different names")
        config = {"mode": "test", "tests_module": tests.stem, "test_id": test_id}
        files = {submission.name: submission.read_bytes(), tests.name: tests_source}
        input_hash = self._input_hash(tests_source)
        return self._cached_run(files, submission, test_id, input_hash, config, driver="")

    def run_code(self, submission: Path, driver: str, stdin: str = "") -> RunResult:
        """Run a custom driver snippet (which may import the submission) with given stdin."""
        config = {"mode": "code"}
        files = {submission.name: submission.read_bytes()}
        input_hash = self._input_hash(f"{driver}\0{stdin}".encode())
        return self._cached_run(
            files, submission, CUSTOM_TEST_ID, input_hash, config, driver=driver, stdin=stdin
        )

    def _input_hash(self, data: bytes) -> str:
        limits = json.dumps(asdict(self.limits), sort_keys=True).encode()
        return sha256_bytes(b"\0".join((data, limits, HARNESS_HASH.encode())))

    def _cached_run(
        self,
        files: dict[str, bytes],
        submission: Path,
        test_id: str,
        input_hash: str,
        config: dict[str, Any],
        driver: str,
        stdin: str = "",
    ) -> RunResult:
        key = CacheKey(sha256_bytes(files[submission.name]), test_id, input_hash)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return hit
        outcome = self._execute(files, config, driver, stdin)
        result = to_result(outcome, None if test_id == CUSTOM_TEST_ID else test_id)
        # A wall-clock timeout can come from a busy machine (e.g. vLLM sharing the CPU), so
        # it is not cached; CPU-time limits measure the code itself and are cached.
        if self.cache is not None and not outcome.timed_out:
            self.cache.put(key, result)
        return result

    def _execute(
        self, files: dict[str, bytes], config: dict[str, Any], driver: str, stdin: str
    ) -> Outcome:
        with tempfile.TemporaryDirectory(prefix="sandbox-", ignore_cleanup_errors=True) as tmp:
            work, control = Path(tmp, "work"), Path(tmp, "control")
            work.mkdir()
            control.mkdir()
            for name, content in files.items():
                (work / name).write_bytes(content)
            nonce = secrets.token_hex(16)
            child_config = {**config, "nonce": nonce, "limits": self.limits.child_config()}
            (control / "config.json").write_text(json.dumps(child_config), encoding="utf-8")
            (control / "driver.py").write_text(driver, encoding="utf-8")
            returncode, timed_out, runtime = self._spawn(work, control, stdin)
            return Outcome(
                returncode=returncode,
                timed_out=timed_out,
                report=read_report(control / "result.json", nonce),
                stdout=read_capped(control / "stdout.txt", self.limits.max_output_bytes),
                stderr=read_capped(control / "stderr.txt", self.limits.max_output_bytes),
                runtime_s=runtime,
            )

    def _spawn(self, work: Path, control: Path, stdin: str) -> tuple[int | None, bool, float]:
        cmd = [self.python, "-s", "-B", str(HARNESS), str(control)]
        with open(control / "stdout.txt", "wb") as out, open(control / "stderr.txt", "wb") as err:
            start = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                cwd=work,
                env=child_env(),
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                start_new_session=IS_POSIX,
            )
            try:
                proc.communicate(stdin.encode("utf-8"), timeout=self.limits.wall_seconds)
                timed_out = False
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                proc.wait()
                timed_out = True
            return proc.returncode, timed_out, time.perf_counter() - start


def function_names(source: bytes) -> set[str]:
    """Top-level function names in a (trusted) tests file, found without importing it."""
    tree = ast.parse(source)
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def child_env() -> dict[str, str]:
    env = {"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    if not IS_POSIX and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]  # required by Python on Windows
    return env


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    try:
        if IS_POSIX:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def read_capped(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = f.read(max_bytes + 1)
    text = data[:max_bytes].decode("utf-8", errors="replace")
    return text + TRUNCATION_NOTE if len(data) > max_bytes else text


def read_report(path: Path, nonce: str) -> dict[str, Any] | None:
    """The harness's result, or None if missing, malformed or not written by this run."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(report, dict) or report.get("nonce") != nonce:
        return None
    return report


def classify(outcome: Outcome) -> ErrorType | None:
    if outcome.timed_out:
        return "timeout"
    if outcome.report is not None:
        status = outcome.report.get("status")
        if status == "passed":
            return None
        return status if status in REPORTED_STATUSES else "crashed"
    if IS_POSIX and outcome.returncode in (-signal.SIGKILL, -signal.SIGXCPU):
        return "timeout"  # killed by the CPU-time limit
    if "MemoryError" in outcome.stderr:
        return "memory_limit"
    return "crashed"


def to_result(outcome: Outcome, test_id: str | None) -> RunResult:
    error_type = classify(outcome)
    return RunResult(
        passed=error_type is None,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        error_type=error_type,
        runtime_s=outcome.runtime_s,
        test_id=test_id,
    )
