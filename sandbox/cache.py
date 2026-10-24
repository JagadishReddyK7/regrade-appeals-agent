"""On-disk cache of sandbox results keyed by (submission hash, test id, input hash)."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sandbox.result import RunResult


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    submission_hash: str
    test_id: str
    input_hash: str

    def filename(self) -> str:
        joined = "\0".join((self.submission_hash, self.test_id, self.input_hash))
        return sha256_bytes(joined.encode("utf-8")) + ".json"


class ResultCache:
    """One JSON file per key. Results are reused as-is, including the original runtime."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, key: CacheKey) -> RunResult | None:
        path = self.root / key.filename()
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("key") != asdict(key):
            return None
        return RunResult.model_validate(record["result"])

    def put(self, key: CacheKey, result: RunResult) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"key": asdict(key), "result": result.model_dump()}
        path = self.root / key.filename()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(path)
