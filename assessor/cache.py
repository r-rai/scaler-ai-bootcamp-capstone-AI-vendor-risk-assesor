"""
cache.py — every cache in the system, in one SQLite file (runs/cache/cache.db).

Caching here is deliberate about WHAT may be reused, because a wrong cache hit in this product is a wrong
compliance answer:

  kind          key                                   reuse rule
  ------------  ------------------------------------  -----------------------------------------------------
  embeddings    sha256(embedder + text)               always safe: same model + same text = same vector
  search        provider + query                      24 h; bypassed by a re-review (refresh)
  page          URL                                   7 days, stores fetched_at; bypassed by a re-review
  llm           sha256(model + full prompt)           content-addressed: the prompt contains its sources, so an
                                                      answer can only be reused for exactly the same inputs
  semantic      meaning of a POLICY question          see SemanticCache below; never used for assessments

What is never cached: verdicts and reports. The matcher is deterministic and cheap; an assessment is always
recomputed from evidence, and cached evidence carries its fetch date into the freshness maths.

Cached page text is still untrusted: the injection scan and identity gate run on every read, cached or not.

The semantic cache follows the Production RAG course (rag_core/cache.py): scope by permission and index
version, keep the threshold high. Two additions for this domain:
  * the scope also contains the prompt-set hash and the answering model, and
  * a hit requires the same "anchors" (data tiers, clause IDs, vendor names, numbers) as well as a close
    embedding, because "Who approves a T2 tool?" and "Who approves a T3 tool?" are nearly the same vector
    and have different answers.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from . import observability as obs
from .config import RUNS_DIR, SETTINGS

CACHE_PATH = RUNS_DIR / "cache" / "cache.db"
LLM_TTL_S = 30 * 86400

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS kv (
        ns TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, created REAL NOT NULL, expires REAL,
        tag TEXT NOT NULL DEFAULT '', hits INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (ns, key))""",
    "CREATE INDEX IF NOT EXISTS kv_tag ON kv (tag)",
    "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL, created REAL NOT NULL)",
    """CREATE TABLE IF NOT EXISTS semantic (
        id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, anchors TEXT NOT NULL, question TEXT NOT NULL,
        vec BLOB NOT NULL, value TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0)""",
    "CREATE INDEX IF NOT EXISTS semantic_scope ON semantic (scope, anchors)",
]


def sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()


class Cache:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        for stmt in _SCHEMA:
            self._conn.execute(stmt)
        self.enabled = True

    # ---- key/value -------------------------------------------------------------
    def get(self, ns: str, key: str, *, run_id: str = "") -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value, expires FROM kv WHERE ns = ? AND key = ?", (ns, key)).fetchone()
            hit = bool(row) and (row[1] is None or row[1] > time.time())
            if hit:
                self._conn.execute("UPDATE kv SET hits = hits + 1 WHERE ns = ? AND key = ?", (ns, key))
        obs.on_trace_event(run_id, "cache", {"cache": ns, "hit": hit})
        return json.loads(row[0]) if hit else None

    def put(self, ns: str, key: str, value: Any, *, ttl: float | None = None, tag: str = "") -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (ns, key, value, created, expires, tag, hits) VALUES (?, ?, ?, ?, ?, ?, 0)",
                (ns, key, json.dumps(value, default=str, ensure_ascii=False), now, now + ttl if ttl else None, tag.lower()))

    # ---- vectors ---------------------------------------------------------------
    def get_vectors(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 400):
                chunk = list(keys[i:i + 400])
                q = "SELECT key, vec FROM vectors WHERE key IN (%s)" % ",".join("?" * len(chunk))
                for k, blob in self._conn.execute(q, chunk):
                    out[k] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_vectors(self, items: dict[str, np.ndarray]) -> None:
        now = time.time()
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO vectors (key, dim, vec, created) VALUES (?, ?, ?, ?)",
                                   [(k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes(), now) for k, v in items.items()])

    # ---- semantic (policy questions only) --------------------------------------
    def semantic_get(self, scope: str, anchors: str, q_vec: np.ndarray, threshold: float) -> Optional[dict]:
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, question, vec, value FROM semantic WHERE scope = ? AND anchors = ? AND expires > ?",
                (scope, anchors, now)).fetchall()
        best, best_sim = None, -1.0
        for rid, question, blob, value in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape != q_vec.shape:
                continue
            sim = float(v @ q_vec)
            if sim > best_sim:
                best, best_sim = (rid, question, value), sim
        if best is None or best_sim < threshold:
            return None
        with self._lock:
            self._conn.execute("UPDATE semantic SET hits = hits + 1 WHERE id = ?", (best[0],))
        return {"similarity": round(best_sim, 4), "matched_question": best[1], "value": json.loads(best[2])}

    def semantic_put(self, scope: str, anchors: str, question: str, q_vec: np.ndarray, value: Any, ttl: float) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute("DELETE FROM semantic WHERE scope = ? AND question = ?", (scope, question))
            self._conn.execute(
                "INSERT INTO semantic (scope, anchors, question, vec, value, created, expires) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (scope, anchors, question, np.asarray(q_vec, dtype=np.float32).tobytes(),
                 json.dumps(value, default=str, ensure_ascii=False), now, now + ttl))

    # ---- admin -------------------------------------------------------------------
    def purge(self, *, ns: str | None = None, tag: str | None = None, expired_only: bool = False) -> int:
        n = 0
        now = time.time()
        with self._lock:
            if expired_only:
                n += self._conn.execute("DELETE FROM kv WHERE expires IS NOT NULL AND expires <= ?", (now,)).rowcount
                n += self._conn.execute("DELETE FROM semantic WHERE expires <= ?", (now,)).rowcount
            elif tag:
                n += self._conn.execute("DELETE FROM kv WHERE tag = ?" + (" AND ns = ?" if ns else ""),
                                        (tag.lower(), ns) if ns else (tag.lower(),)).rowcount
            elif ns == "semantic":
                n += self._conn.execute("DELETE FROM semantic").rowcount
            elif ns == "embeddings":
                n += self._conn.execute("DELETE FROM vectors").rowcount
            elif ns:
                n += self._conn.execute("DELETE FROM kv WHERE ns = ?", (ns,)).rowcount
            else:
                for table in ("kv", "semantic", "vectors"):
                    n += self._conn.execute(f"DELETE FROM {table}").rowcount
        return n

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            out = {ns: {"entries": n, "hits": hits or 0, "bytes": size or 0}
                   for ns, n, hits, size in self._conn.execute("SELECT ns, COUNT(*), SUM(hits), SUM(LENGTH(value)) FROM kv GROUP BY ns")}
            n, hits, size = self._conn.execute("SELECT COUNT(*), SUM(hits), SUM(LENGTH(value)) FROM semantic").fetchone()
            out["semantic"] = {"entries": n, "hits": hits or 0, "bytes": size or 0}
            n, size = self._conn.execute("SELECT COUNT(*), SUM(LENGTH(vec)) FROM vectors").fetchone()
            out["embeddings"] = {"entries": n, "hits": 0, "bytes": size or 0}
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class NullCache:
    """Used when caching is off (tests, VRA_CACHE=0): every lookup misses, every write is dropped."""

    enabled = False

    def get(self, *a: Any, **k: Any) -> Any:
        return None

    def put(self, *a: Any, **k: Any) -> None:
        return None

    def get_vectors(self, keys: Sequence[str]) -> dict:
        return {}

    def put_vectors(self, items: dict) -> None:
        return None

    def semantic_get(self, *a: Any, **k: Any) -> None:
        return None

    def semantic_put(self, *a: Any, **k: Any) -> None:
        return None

    def purge(self, **k: Any) -> int:
        return 0

    def stats(self) -> dict:
        return {}


_CACHE: Cache | NullCache | None = None
_CACHE_LOCK = threading.Lock()


def get_cache() -> Cache | NullCache:
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            try:
                _CACHE = Cache() if SETTINGS.cache_enabled else NullCache()
            except (OSError, sqlite3.Error):   # read-only filesystem: run without a cache rather than fail
                _CACHE = NullCache()
        return _CACHE


def set_cache(cache: Cache | NullCache | None) -> None:
    """Swap the process cache (tests, admin script)."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = cache


# ---------------------------------------------------------------------------
# Semantic cache for policy answers
# ---------------------------------------------------------------------------

_TIER_RE = re.compile(r"\bT[1-3]\b|\btier\s*[1-3]\b", re.I)
_CLAUSE_RE = re.compile(r"\b[A-Z]{1,5}(?:-[\dA-Z]+)+(?:\.\d+)*\b")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_NEG_RE = re.compile(r"\b(not|never|no|without|except|unless|cannot|can't|isn't|aren't|don't|doesn't)\b", re.I)


def question_anchors(question: str, vendors: Iterable[str] = ()) -> str:
    """The parts of a question that an embedding blurs but that change the answer. Two questions may share a
    cached answer only if these are identical."""
    q = question
    tiers = sorted({re.sub(r"(?i)tier\s*", "T", m.group(0)).upper() for m in _TIER_RE.finditer(q)})
    clauses = sorted({m.group(0).upper() for m in _CLAUSE_RE.finditer(q)})
    low = q.lower()
    names = sorted({v.lower() for v in vendors if v and (v.lower() in low or v.split()[0].lower() in re.findall(r"[a-z0-9]+", low))})
    numbers = sorted(set(_NUM_RE.findall(re.sub(_TIER_RE, " ", q))))
    return json.dumps({"tiers": tiers, "clauses": clauses, "vendors": names, "numbers": numbers,
                       "negated": bool(_NEG_RE.search(q))}, sort_keys=True)


def semantic_scope(user_groups: Iterable[str], policy_hash: str, prompt_hash: str, model: str) -> str:
    """Permission + index version + prompt version + model. A policy edit, a different ACL group, a prompt
    change or a model change all produce a different scope, so old answers simply stop matching."""
    return "|".join([",".join(sorted(set(user_groups) | {"public"})), policy_hash, prompt_hash, model])
