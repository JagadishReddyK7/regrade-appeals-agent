import os
from pathlib import Path

import pytest

from sandbox.cache import ResultCache
from sandbox.runner import Limits, SandboxRunner

posix_only = pytest.mark.skipif(os.name != "posix", reason="resource limits need POSIX")

TESTS_SOURCE = """\
from mathlib import add

def test_add():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-2, -3) == -5
"""

FAST_LIMITS = Limits(cpu_seconds=2, memory_mb=256, wall_seconds=4.0)


def write(folder: Path, name: str, source: str) -> Path:
    path = folder / name
    path.write_text(source, encoding="utf-8")
    return path


@pytest.fixture
def tests_file(tmp_path: Path) -> Path:
    return write(tmp_path, "test_mathlib.py", TESTS_SOURCE)


@pytest.fixture
def runner() -> SandboxRunner:
    return SandboxRunner(limits=FAST_LIMITS)


def run_add(runner: SandboxRunner, tmp_path: Path, tests_file: Path, body: str):
    submission = write(tmp_path, "mathlib.py", f"def add(a, b):\n{body}\n")
    return runner.run_test(submission, tests_file, "test_add")


def test_passing_code(runner: SandboxRunner, tmp_path: Path, tests_file: Path) -> None:
    result = run_add(runner, tmp_path, tests_file, "    return a + b")
    assert result.passed
    assert result.error_type is None
    assert result.test_id == "test_add"
    assert result.runtime_s > 0


def test_failing_code(runner: SandboxRunner, tmp_path: Path, tests_file: Path) -> None:
    result = run_add(runner, tmp_path, tests_file, "    return a - b")
    assert not result.passed
    assert result.error_type == "test_failed"
    assert "AssertionError" in result.stderr


def test_exception_in_code(runner: SandboxRunner, tmp_path: Path, tests_file: Path) -> None:
    result = run_add(runner, tmp_path, tests_file, "    return a / 0")
    assert result.error_type == "exception"
    assert "ZeroDivisionError" in result.stderr


def test_infinite_loop_times_out(runner: SandboxRunner, tmp_path: Path, tests_file: Path) -> None:
    result = run_add(runner, tmp_path, tests_file, "    while True:\n        pass")
    assert not result.passed
    assert result.error_type == "timeout"
    assert result.runtime_s < FAST_LIMITS.wall_seconds + 3


def test_sleeping_code_hits_wall_clock(tmp_path: Path, tests_file: Path) -> None:
    runner = SandboxRunner(limits=Limits(wall_seconds=1.0))
    body = "    import time\n    time.sleep(30)"
    assert run_add(runner, tmp_path, tests_file, body).error_type == "timeout"


@posix_only
def test_cpu_limit_stops_busy_loop_before_wall_clock(tmp_path: Path, tests_file: Path) -> None:
    runner = SandboxRunner(limits=Limits(cpu_seconds=1, wall_seconds=20.0))
    result = run_add(runner, tmp_path, tests_file, "    while True:\n        pass")
    assert result.error_type == "timeout"
    assert result.runtime_s < 10


@posix_only
def test_memory_blowup(runner: SandboxRunner, tmp_path: Path, tests_file: Path) -> None:
    result = run_add(runner, tmp_path, tests_file, "    x = bytearray(2 * 1024**3)\n    return 5")
    assert not result.passed
    assert result.error_type == "memory_limit"


@pytest.mark.parametrize(
    "body",
    [
        "    import socket\n    socket.create_connection(('example.com', 80))",
        "    import urllib.request\n    urllib.request.urlopen('http://example.com')",
        "    import importlib\n    importlib.import_module('_socket')",
    ],
)
def test_socket_use_is_blocked(
    runner: SandboxRunner, tmp_path: Path, tests_file: Path, body: str
) -> None:
    result = run_add(runner, tmp_path, tests_file, body)
    assert not result.passed
    assert result.error_type == "network_blocked"


def test_run_code_with_custom_input(runner: SandboxRunner, tmp_path: Path) -> None:
    submission = write(tmp_path, "mathlib.py", "def add(a, b):\n    return a + b\n")
    driver = "from mathlib import add\na, b = map(int, input().split())\nprint(add(a, b))"
    result = runner.run_code(submission, driver, stdin="4 5\n")
    assert result.passed
    assert result.stdout.strip() == "9"
    assert result.test_id is None


def test_working_directory_is_temporary(runner: SandboxRunner, tmp_path: Path) -> None:
    submission = write(tmp_path, "mathlib.py", "")
    driver = "import os\nprint(os.getcwd())\nprint(sorted(os.listdir('.')))"
    result = runner.run_code(submission, driver)
    cwd, listing = result.stdout.splitlines()
    assert Path(cwd) != tmp_path
    assert listing == "['mathlib.py']"


def test_output_is_truncated(tmp_path: Path) -> None:
    runner = SandboxRunner(limits=Limits(max_output_bytes=100))
    submission = write(tmp_path, "mathlib.py", "")
    result = runner.run_code(submission, "print('x' * 1000)")
    assert result.stdout.endswith("[output truncated]")
    assert len(result.stdout) < 200


def test_forged_result_file_is_ignored(runner: SandboxRunner, tmp_path: Path) -> None:
    driver = (
        "import json, os, sys\n"
        "path = os.path.join(sys.argv[1], 'result.json')\n"
        "open(path, 'w').write(json.dumps({'nonce': 'guess', 'status': 'passed'}))\n"
        "os._exit(0)"
    )
    submission = write(tmp_path, "mathlib.py", "")
    result = runner.run_code(submission, driver)
    assert not result.passed
    assert result.error_type == "crashed"


def test_unknown_test_id_rejected(runner: SandboxRunner, tmp_path: Path, tests_file: Path) -> None:
    submission = write(tmp_path, "mathlib.py", "def add(a, b):\n    return a + b\n")
    with pytest.raises(ValueError, match="unknown test"):
        runner.run_test(submission, tests_file, "test_missing")


def test_results_are_cached(tmp_path: Path, tests_file: Path) -> None:
    cache = ResultCache(tmp_path / "cache")
    runner = SandboxRunner(cache=cache, limits=FAST_LIMITS)
    first = run_add(runner, tmp_path, tests_file, "    return a + b")
    runner._execute = None  # type: ignore[assignment]  # any real run would now fail
    second = run_add(runner, tmp_path, tests_file, "    return a + b")
    assert second == first


def test_cache_key_depends_on_submission(tmp_path: Path, tests_file: Path) -> None:
    runner = SandboxRunner(cache=ResultCache(tmp_path / "cache"), limits=FAST_LIMITS)
    assert run_add(runner, tmp_path, tests_file, "    return a + b").passed
    assert not run_add(runner, tmp_path, tests_file, "    return a - b").passed
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2
