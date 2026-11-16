"""
errors.py — durable record of runtime errors so the bug-fix agent can find them later.

`record_error()` appends one JSON line to runs/issues/errors.jsonl. It NEVER raises: error reporting must
not become a new source of errors. Secrets are masked before anything is written.
"""

from __future__ import annotations

import json
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RUNS_DIR

ISSUES_DIR = RUNS_DIR / "issues"
ERRORS_FILE = ISSUES_DIR / "errors.jsonl"
MAX_BYTES = 5 * 1024 * 1024
_LOCK = threading.Lock()

_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.I)
_KEY_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16}|gsk_[A-Za-z0-9]{20,}|tvly-[A-Za-z0-9-]{10,})")
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}")


def mask_secrets(text: str) -> str:
    """Mask API-key-like strings and the literal values of any *_KEY/*_TOKEN/*_SECRET env vars."""
    if not text:
        return text
    for name, val in os.environ.items():
        if val and len(val) >= 8 and _SECRET_ENV.search(name):
            text = text.replace(val, f"[{name} REDACTED]")
    text = _KEY_RE.sub("[SECRET REDACTED]", text)
    return _BEARER_RE.sub(lambda m: m.group(1) + "[SECRET REDACTED]", text)


def record_error(source: str, exc: BaseException, context: dict[str, Any] | None = None,
                 path: Path | None = None) -> None:
    try:
        target = path or ERRORS_FILE
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "type": type(exc).__name__,
            "message": mask_secrets(str(exc))[:1000],
            "traceback": mask_secrets("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))[-6000:],
            "context": json.loads(mask_secrets(json.dumps(context or {}, default=str))),
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with _LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size > MAX_BYTES:
                target.replace(target.with_suffix(".jsonl.1"))
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(line)  # single write() so concurrent API threads can't interleave
    except Exception:  # noqa: BLE001
        pass


def read_errors(path: Path | None = None) -> list[dict]:
    target = path or ERRORS_FILE
    out: list[dict] = []
    if not target.exists():
        return out
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # corrupt line: skip
    return out
