"""A content-addressed cache for agent calls: one JSON file per key, written atomically.

The key covers the role, the frozen prompt version, the model and the payload, so a prompt or model change can never reuse a
stale answer. `salt` draws an independent sample for the SAME inputs (noise measurements); an empty salt adds nothing, so a
salted key can never collide with an unsalted one."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from .config import CACHE_DIR


class CallCache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else CACHE_DIR
        self.hits = self.misses = 0
        self._lock = threading.Lock()

    def key(self, role: str, prompt_version: str, model: str, payload: str, salt: str = "") -> str:
        h = hashlib.sha256()
        for part in (role, prompt_version, model, payload) + (("salt:" + salt,) if salt else ()):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> dict | None:
        try:
            record = json.loads((self.root / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):                        # absent or unreadable (e.g. torn by a crash): a miss, never an error
            record = None
        with self._lock:
            if record is None:
                self.misses += 1
            else:
                self.hits += 1
        return record

    def put(self, key: str, record: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        tmp = path.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")    # unique per writer: parallel workers may store the same key
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
