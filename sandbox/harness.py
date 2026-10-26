"""Child-process entry point for sandboxed runs. Never imported by the parent process.

Usage (by sandbox.runner only): python -s -B harness.py <control_dir>

The harness reads its config from <control_dir>/config.json and deletes it, applies
resource limits (POSIX only), blocks networking and process creation, runs the student's
code from the current working directory, and writes a result JSON file tagged with a
per-run nonce.
It deliberately imports nothing that pulls in `socket` or `subprocess`.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

NETWORK_MODULES = frozenset({"socket", "_socket", "ssl", "_ssl"})
PROCESS_MODULES = frozenset({"subprocess", "_posixsubprocess", "multiprocessing"})
PROCESS_FUNCTIONS = (
    "system", "popen", "fork", "forkpty", "startfile",
    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "posix_spawn", "posix_spawnp",
)  # fmt: skip


class SandboxBlockedError(ImportError):
    def __init__(self, kind: str, what: str) -> None:
        super().__init__(f"{kind} access is disabled in the sandbox ({what!r})")
        self.kind = kind  # "network" or "process"


class ModuleBlocker:
    """Meta-path finder that refuses to import networking and process modules."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        root = name.split(".")[0]
        if root in NETWORK_MODULES:
            raise SandboxBlockedError("network", name)
        if root in PROCESS_MODULES:
            raise SandboxBlockedError("process", name)
        return None


def blocked_function(name: str) -> Any:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise SandboxBlockedError("process", f"os.{name}")

    return refuse


def install_blocks() -> None:
    """Block network and process creation. Must run after the harness's own imports."""
    for name in NETWORK_MODULES | PROCESS_MODULES:
        sys.modules.pop(name, None)
    sys.meta_path.insert(0, ModuleBlocker())
    for name in PROCESS_FUNCTIONS:
        if hasattr(os, name):
            setattr(os, name, blocked_function(name))


def apply_limits(cpu_seconds: int, memory_mb: int, fsize_bytes: int) -> None:
    try:
        import resource
        import signal
    except ImportError:  # Windows: only the parent's wall-clock timeout applies
        return
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # oversized writes fail instead of killing
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    memory = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def blocked_kind(exc: BaseException | None) -> str | None:
    """"network" or "process" if exc was caused by a sandbox block, else None."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, SandboxBlockedError):
            return exc.kind
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def invoke(config: dict[str, Any], control: Path) -> None:
    if config["mode"] == "test":
        module = __import__(config["tests_module"])
        getattr(module, config["test_id"])()
    else:
        driver = (control / "driver.py").read_text(encoding="utf-8")
        exec(compile(driver, "<driver>", "exec"), {"__name__": "__main__"})


def summarise(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def run_student_code(config: dict[str, Any], control: Path) -> tuple[str, str]:
    """Run the requested test or driver; return (status, exception summary)."""
    try:
        invoke(config, control)
    except MemoryError:
        return "memory_limit", "MemoryError"
    except AssertionError as exc:
        traceback.print_exc()
        return ("test_failed" if config["mode"] == "test" else "exception"), summarise(exc)
    except BaseException as exc:  # includes SystemExit and KeyboardInterrupt from student code
        traceback.print_exc()
        kind = blocked_kind(exc)
        return (f"{kind}_blocked" if kind else "exception"), summarise(exc)
    return "passed", ""


def main(argv: list[str]) -> int:
    control = Path(argv[1])
    config_path = control / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_path.unlink()
    nonce = config["nonce"]
    limits = config["limits"]
    apply_limits(limits["cpu_seconds"], limits["memory_mb"], limits["fsize_bytes"])
    install_blocks()
    sys.path[0] = os.getcwd()
    status, detail = run_student_code(config, control)
    sys.stdout.flush()
    sys.stderr.flush()
    result = {"nonce": nonce, "status": status, "detail": detail}
    (control / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
