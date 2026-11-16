"""
auth.py — who is calling, decided on the server.

Project rule: "Groups come from identity on the server, never from the request body." A caller presents
a token (`X-API-Token` or `Authorization: Bearer`); the token maps to a user, the ACL groups used by
policy retrieval, and the approver roles that user may sign in (AI-5.1).

Token directory
  VRA_TOKENS_FILE points at a JSON file:
      {"tokens": [{"sha256": "<hex of sha256(token)>", "user": "...", "groups": [...], "roles": [...]}]}
  Only hashes are stored, so reading the file does not reveal a usable token.
  Create entries with:  python scripts/manage_tokens.py add --user ... --groups ... --roles ...

Modes (VRA_ENV)
  dev   (default) the demo tokens below also work, and a call without a token is accepted as an anonymous
        caller using the groups in the request body. This keeps the course demo and offline tests simple.
  prod  demo tokens are not loaded; every call needs a valid token; request-body groups are ignored.

This is file-based token auth for a single-machine deployment. It is not SSO; put the service behind
your identity provider before exposing it beyond localhost.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .config import SETTINGS

# Approver roles per data tier (AI-5.1 "Who approves"). T3 needs BOTH roles, signed by two people.
APPROVERS: dict[str, list[str]] = {
    "T1": ["Team lead"],
    "T2": ["Head of Information Security"],
    "T3": ["Head of Information Security", "DPO"],
}
PRIVILEGED_GROUPS = frozenset({"security", "procurement"})   # may read any run, history, stats and the audit log

# Demo identities (dev mode only). The users are fictional; `.example` is a reserved domain.
DEMO_TOKENS: dict[str, dict] = {
    "tok-public": {"user": "guest", "groups": ["public"], "roles": []},
    "tok-employee": {"user": "ravi.k@nimbumart.example", "groups": ["public", "employees"], "roles": []},
    "tok-teamlead": {"user": "team.lead@nimbumart.example", "groups": ["public", "employees"], "roles": ["Team lead"]},
    "tok-security": {"user": "sec.lead@nimbumart.example", "groups": ["public", "employees", "security"],
                     "roles": ["Head of Information Security"]},
    "tok-procurement": {"user": "proc.lead@nimbumart.example", "groups": ["public", "employees", "procurement"], "roles": []},
    "tok-dpo": {"user": "dpo@nimbumart.example", "groups": ["public", "employees", "security", "procurement"], "roles": ["DPO"]},
}


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class Identity:
    user: str
    groups: tuple[str, ...] = ("public",)
    roles: tuple[str, ...] = ()
    authenticated: bool = False

    @property
    def privileged(self) -> bool:
        return bool(PRIVILEGED_GROUPS & set(self.groups))

    def can_read_run(self, requested_by: str) -> bool:
        """A run is visible to whoever requested it and to Security / Procurement."""
        if not SETTINGS.prod and not self.authenticated:
            return True          # dev convenience: the anonymous demo caller sees local runs
        return self.privileged or (bool(requested_by) and requested_by == self.user)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenDirectory:
    """sha256(token) -> Identity. Reloads the file when it changes, so a revoked token stops working at once."""

    def __init__(self, path: str | None = None, *, include_demo: bool | None = None):
        self._path = Path(path) if path else None
        self._include_demo = include_demo
        self._state: tuple = ()
        self._entries: dict[str, Identity] = {}
        self._lock = threading.Lock()

    def _load(self) -> None:
        path = self._path or (Path(SETTINGS.tokens_file) if SETTINGS.tokens_file else None)
        demo = self._include_demo if self._include_demo is not None else not SETTINGS.prod
        mtime = path.stat().st_mtime if path and path.exists() else 0.0
        state = (str(path), mtime, demo)     # reload when the file, its content or the mode changes
        if state == self._state:
            return
        entries: dict[str, Identity] = {}
        if demo:
            for tok, e in DEMO_TOKENS.items():
                entries[hash_token(tok)] = Identity(e["user"], tuple(e["groups"]), tuple(e["roles"]), True)
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for e in data.get("tokens", []):
                digest = str(e.get("sha256", "")).lower()
                if len(digest) != 64 or not e.get("user"):
                    continue
                groups = tuple(dict.fromkeys(["public", *e.get("groups", [])]))
                entries[digest] = Identity(e["user"], groups, tuple(e.get("roles", [])), True)
        self._entries, self._state = entries, state

    def lookup(self, token: str) -> Optional[Identity]:
        with self._lock:
            self._load()
            return self._entries.get(hash_token(token))

    def __len__(self) -> int:
        with self._lock:
            self._load()
            return len(self._entries)


DIRECTORY = TokenDirectory()


def extract_token(x_api_token: Optional[str], authorization: Optional[str]) -> Optional[str]:
    if x_api_token:
        return x_api_token.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def authenticate(token: Optional[str], fallback_groups: Iterable[str] = ("public", "employees"),
                 directory: TokenDirectory | None = None) -> Identity:
    """Resolve the caller. Raises AuthError(401) for a bad token, or for no token in prod."""
    directory = directory if directory is not None else DIRECTORY   # an empty directory is still a directory
    if token:
        ident = directory.lookup(token)
        if ident is None:
            raise AuthError(401, "Invalid API token")
        return ident
    if SETTINGS.prod:
        raise AuthError(401, "An API token is required (X-API-Token or Authorization: Bearer).")
    return Identity("api", tuple(fallback_groups), (), False)


# ---------------------------------------------------------------------------
# Rate limiting: a token bucket per caller, in process. Enough for one machine; use a gateway beyond that.
# ---------------------------------------------------------------------------

@dataclass
class _Bucket:
    tokens: float
    updated: float = field(default_factory=time.monotonic)


class RateLimiter:
    def __init__(self, per_minute: int | None = None):
        self.per_minute = per_minute
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        rate = self.per_minute if self.per_minute is not None else SETTINGS.rate_per_min
        if rate <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            b = self._buckets.setdefault(key, _Bucket(tokens=float(rate), updated=now))
            b.tokens = min(float(rate), b.tokens + (now - b.updated) * rate / 60.0)
            b.updated = now
            if b.tokens >= cost:
                b.tokens -= cost
                return True
            return False


LIMITER = RateLimiter()
