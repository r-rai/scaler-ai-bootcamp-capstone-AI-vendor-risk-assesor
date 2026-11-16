"""
provenance.py — what produced a report.

An auditor's question is "which policies, rules, prompts, models and code gave this verdict, and would
the same inputs give it again?". Every report carries the answer: content hashes of the policy library,
the requirements file and the prompt set, the model per role, the retrieval backends and the code commit.
Two reports with the same stamp and the same evidence must reach the same verdicts.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

from .config import DATA_DIR, POLICY_DIR, REQUIREMENTS_FILE, ROOT, SETTINGS


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


@lru_cache(maxsize=4)
def policy_corpus_hash(policy_dir: str = str(POLICY_DIR)) -> str:
    """Changes whenever any policy document or its metadata (ACL, status, version) changes."""
    d = Path(policy_dir)
    return hash_files([p for p in d.iterdir() if p.is_file() and p.suffix in (".md", ".json")])


@lru_cache(maxsize=1)
def requirements_hash() -> str:
    return _sha(REQUIREMENTS_FILE.read_bytes())


@lru_cache(maxsize=1)
def source_types_hash() -> str:
    p = DATA_DIR / "source_types.yaml"
    return _sha(p.read_bytes()) if p.exists() else ""


@lru_cache(maxsize=1)
def prompt_set_hash() -> str:
    """One hash over every prompt the pipeline can send. A prompt edit changes it, which also invalidates caches."""
    from . import graders, guardrails, rag_core, router, vendor_research

    prompts = [
        router.ROUTER_BACKSTORY_SEMANTIC, router.ROUTER_RULES, graders.RELEVANCE_SYSTEM, graders.GROUNDED_SYSTEM,
        guardrails.INJECTION_CLASSIFIER_SYSTEM, rag_core.POLICY_SYSTEM_PROMPT, vendor_research.PLAN_CREATOR_SYSTEM,
        vendor_research.EXTRACT_SYSTEM, vendor_research.IDENTITY_SYSTEM,
        getattr(vendor_research, "QUERY_SYSTEM", ""), getattr(vendor_research, "DOMAIN_SYSTEM", ""),
    ]
    return _sha("\x1e".join(prompts).encode("utf-8"))


@lru_cache(maxsize=1)
def code_version() -> str:
    """Git commit of the running code, read from .git without spawning a process. VRA_GIT_SHA wins (set in Docker)."""
    env = os.environ.get("VRA_GIT_SHA", "").strip()
    if env:
        return env[:12]
    try:
        git = ROOT / ".git"
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = git / head.split(" ", 1)[1]
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()[:12]
            for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
                if line.endswith(head.split(" ", 1)[1]):
                    return line.split(" ")[0][:12]
            return "unknown"
        return head[:12]
    except OSError:
        return "unknown"


def models_in_use() -> dict[str, str]:
    out: dict[str, str] = {}
    for role in ("main", "small", "extract"):
        provider = SETTINGS.provider_for(role)
        if provider == "mock":
            out[role] = "mock"
            continue
        try:
            out[role] = f"{provider}:{SETTINGS.resolved_llm(role=role, provider=provider)['model']}"
        except ValueError:
            out[role] = f"{provider}:unconfigured"
    return out


def stamp(*, embedder: str = "", reranker: str = "", cache_hits: dict | None = None, extra: dict | None = None) -> dict:
    s = {
        "policy_corpus": policy_corpus_hash(),
        "requirements": requirements_hash(),
        "prompts": prompt_set_hash(),
        "models": models_in_use(),
        "embedder": embedder,
        "reranker": reranker,
        "search_provider": SETTINGS.search_provider,
        "code": code_version(),
    }
    if source_types_hash():
        s["source_types"] = source_types_hash()
    if cache_hits:
        s["cache_hits"] = dict(cache_hits)
    if extra:
        s.update(extra)
    return s
