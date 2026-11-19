"""A content-addressed cache for agent calls: one JSON file per key, written atomically.

The key covers the role, the frozen prompt version, the model and the payload, so a prompt or model change can never reuse a
stale answer. `salt` draws an independent sample for the SAME inputs (noise measurements); an empty salt adds nothing, so a
salted key can never collide with an unsalted one.

The cache is best effort: `put` never raises (a failed write is counted in `put_failures` and only costs a later re-call) and
`get` treats an unreadable entry as a miss."""
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
        self.hits = self.misses = self.put_failures = 0
        self._lock = threading.Lock()

    def key(self, role: str, prompt_version: str, model: str, payload: str, salt: str = "") -> str:
        h = hashlib.sha256()
        for part in (role, prompt_version, model, payload) + (("salt:" + salt,) if salt else ()):
            h.update(part.encode("utf-8", "surrogatepass"))      # same bytes as strict utf-8 for any text without a lone surrogate
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
        tmp = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / f"{key}.json"
            tmp = path.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")    # unique per writer: parallel workers may store the same key
            tmp.write_text(json.dumps(record), encoding="utf-8")        # ASCII output (escapes lone surrogates too), json.loads restores the text
            tmp.replace(path)
        except Exception:                                    # best effort: a cache write must never fail a run that already paid for the call
            with self._lock:
                self.put_failures += 1
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
