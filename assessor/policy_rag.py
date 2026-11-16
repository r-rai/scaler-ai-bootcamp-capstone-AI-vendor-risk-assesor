"""
policy_rag.py — the internal policy library: Nimbu Mart's own documents, searched with access control.

COURSE MAP
  Production RAG course  Module 6  (chunking: we chunk by clause heading so every hit has a clause id)
                         Module 8  (retrieval: hybrid BM25 + dense vectors fused via RRF + cross-encoder rerank)
                         Module 9  (grounded generation: LLM grounded answers with clause citations + extractive fallback)
                         Module 11 (security: ACL lives IN retrieval; archived, draft, untrusted, and injected chunks filtered)

Milestone 1 complete:
  * Hybrid search (BM25 + DenseEmbedder fused with Reciprocal Rank Fusion, k=60)
  * Cross-encoder reranker (`CrossEncoderReranker`) that resolves semantic/table queries like `pq-who-approves`
  * Pre-ranking security filters: ACL (`user_groups`), active status (`status == 'active'`), trust (`trusted`), and prompt-injection quarantine
  * Grounded LLM generation with automatic extractive fallback when running offline (`mock` mode)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import injection
from .config import LOCAL_PROVIDERS, POLICY_DIR
from .rag_core import (
    POLICY_SYSTEM_PROMPT, DenseEmbedder, expand_concepts, get_reranker, linearise_tables, reciprocal_rank_fusion, _stem,
)
from .schemas import PolicyAnswer, PolicyHit

STOPWORDS = set("""a an and are as at be by for from has have how if in is it its of on or our that the this to was we what
when where which who will with you your must may can does do did into than then there these those also any all not no
according say says said""".split())

_CLAUSE_RE = re.compile(r"^([A-Z]{1,5}(?:-[\dA-Z]+)+(?:\.\d+)*)\b\s*(.*)$")  # SEC-4.2, AVL-1, Q-SEC-4.2, IR-1.1


def tokenize(text: str) -> list[str]:
    text = text.lower()
    ids = re.findall(r"\b[a-z]{1,5}-\d+(?:\.\d+)*\b", text)
    words = re.findall(r"[a-z0-9]+", text)
    base = [w for w in words if w not in STOPWORDS and len(w) > 1]
    stems = [_stem(w) for w in base if _stem(w) != w]
    return base + stems + ids


@dataclass
class PolicyChunk:
    doc: str
    title: str
    doc_id: str
    clause_id: str
    heading: str
    text: str
    version: str
    status: str
    trust: str
    access: list[str]
    effective_date: str = ""
    tokens: list[str] = field(default_factory=list)
    injection_flag: bool = False


def chunk_markdown(doc: str, text: str, meta: dict) -> list[PolicyChunk]:
    """One chunk per `###` clause (or per `##` section when a section has no clauses).

    Production RAG course Module 6, "Structure-aware chunking": headings are the natural unit of a
    policy. Each chunk keeps its clause id so a citation can say "SEC-4.2", not "chunk 17".
    """
    chunks: list[PolicyChunk] = []
    section = ""
    cur_head, cur_lines, cur_clause = "", [], ""

    def flush():
        body = "\n".join(l for l in cur_lines).strip()
        if body:
            head = cur_head or section or meta.get("title", doc)
            full_text = f"{head}\n{body}"
            chunks.append(PolicyChunk(
                doc=doc, title=meta.get("title", doc), doc_id=meta.get("doc_id", ""), clause_id=cur_clause,
                heading=head, text=body, version=str(meta.get("version", "")), status=meta.get("status", "active"),
                trust=meta.get("trust", "trusted"), access=list(meta.get("access", ["public"])),
                effective_date=str(meta.get("effective_date", "")),
                tokens=tokenize(full_text),
                injection_flag=injection.is_injection(full_text),
            ))

    for line in text.splitlines():
        if line.startswith("### "):
            flush()
            cur_lines = []
            raw = line[4:].strip()
            m = _CLAUSE_RE.match(raw)
            cur_clause, cur_head = (m.group(1), raw) if m else ("", raw)
        elif line.startswith("## "):
            flush()
            cur_lines, cur_clause, cur_head = [], "", ""
            section = line[3:].strip()
        elif line.startswith("# "):
            flush()
            cur_lines, cur_clause, cur_head = [], "", ""
        else:
            cur_lines.append(line)
    flush()
    return chunks


class BM25:
    """BM25 keyword scorer (Production RAG Module 8)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.n)
        self.df: Counter = Counter()
        for d in docs:
            self.df.update(set(d))
        self.tf = [Counter(d) for d in docs]

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: list[str], idx: int) -> float:
        tf, dl = self.tf[idx], len(self.docs[idx])
        s = 0.0
        for q in query:
            f = tf.get(q, 0)
            if not f:
                continue
            s += self.idf(q) * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


class PolicyLibrary:
    ABSTAIN_BELOW = 0.22  # Reranked / hybrid threshold for the built-in scorer; tuned with eval/run_eval.py
    # With a real cross-encoder the score is sigmoid(logit / 4). On this library, answerable questions score
    # 0.45+ for their best clause and unrelated ones about 0.07; 0.25 (logit -4.4) sits in the gap. Re-check
    # with eval/run_eval.py if the reranker model changes.
    MODEL_ABSTAIN_BELOW = 0.25
    BM25_ABSTAIN_BELOW = 2.0
    PUBLIC_GROUPS = frozenset({"public", "employees"})   # chunks readable by these may be sent to a cloud model

    def __init__(self, policy_dir: Path = POLICY_DIR, *, use_hybrid: bool = True, use_rerank: bool = True):
        self.policy_dir = Path(policy_dir)
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank
        self.meta: dict = json.loads((self.policy_dir / "_metadata.json").read_text(encoding="utf-8"))
        self.chunks: list[PolicyChunk] = []
        for name, m in self.meta.items():
            if name.startswith("_"):
                continue
            p = self.policy_dir / name
            if not p.exists():
                continue
            self.chunks.extend(chunk_markdown(name, p.read_text(encoding="utf-8"), m))
        self.bm25 = BM25([c.tokens for c in self.chunks])
        self.embedder = DenseEmbedder()
        self.reranker = get_reranker()
        # What the models read: tables written out as sentences (see rag_core.linearise_tables). The built-in
        # fallback embedder keeps the raw text it was tuned on.
        self.model_texts = [f"{c.heading}\n{linearise_tables(c.text)}" for c in self.chunks]
        if self.embedder.backend == "builtin":
            chunk_texts = [f"{c.title} > {c.heading}\n{c.text}" for c in self.chunks]
        else:
            chunk_texts = [f"{c.title} > {t}" for c, t in zip(self.chunks, self.model_texts)]
        self.vectors = self.embedder.encode(chunk_texts)

    # ------------------------------------------------------------------ filters
    @staticmethod
    def _allowed(chunk: PolicyChunk, user_groups: Iterable[str]) -> bool:
        groups = set(user_groups) | {"public"}
        return bool(groups & set(chunk.access))

    def search(self, question: str, user_groups: Iterable[str] = ("public",), *, k: int = 5,
               include_archived: bool = False, trusted_only: bool = True) -> list[PolicyHit]:
        """Filter FIRST (ACL, active status, trust, injection quarantine), then hybrid search (BM25 + vectors, RRF) + rerank."""
        q_tokens = tokenize(question)
        allowed_indices: list[int] = []
        for i, c in enumerate(self.chunks):
            if not self._allowed(c, user_groups):
                continue
            if c.status != "active" and not include_archived:
                continue
            if trusted_only and (c.trust != "trusted" or c.injection_flag):
                continue
            allowed_indices.append(i)

        if not allowed_indices:
            return []

        # 1. BM25 scores over allowed chunks
        bm25_scores: dict[int, float] = {}
        for i in allowed_indices:
            c = self.chunks[i]
            s = self.bm25.score(q_tokens, i)
            if c.clause_id and c.clause_id.lower() in question.lower():
                s += 5.0
            if s > 0:
                bm25_scores[i] = s

        if not self.use_hybrid and not self.use_rerank:
            # Pure BM25 baseline mode (used in ablation study)
            ranked = sorted(bm25_scores.items(), key=lambda x: -x[1])[:k]
            return [
                PolicyHit(
                    doc=self.chunks[i].doc, title=self.chunks[i].title, doc_id=self.chunks[i].doc_id,
                    clause_id=self.chunks[i].clause_id, heading=self.chunks[i].heading, text=self.chunks[i].text,
                    score=round(sc, 3), version=self.chunks[i].version, status=self.chunks[i].status,
                    trust=self.chunks[i].trust, access=self.chunks[i].access,
                )
                for i, sc in ranked
            ]

        # 2. Vector similarity over allowed chunks
        q_vec = self.embedder.encode([question])[0]
        vec_scores: dict[int, float] = {}
        for i in allowed_indices:
            sim = float(np.dot(self.vectors[i], q_vec))
            if sim > 0.05:
                vec_scores[i] = sim

        # Guard: if neither BM25 nor vector has any meaningful lexical/concept overlap, return empty
        q_concepts = set(expand_concepts(q_tokens))
        has_any_match = bool(bm25_scores) or any(
            bool(q_concepts & set(expand_concepts(self.chunks[i].tokens))) for i in allowed_indices
        )
        if not has_any_match:
            return []

        bm25_ranked = [i for i, _ in sorted(bm25_scores.items(), key=lambda x: -x[1])]
        vec_ranked = [i for i, _ in sorted(vec_scores.items(), key=lambda x: -x[1])]

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        rrf_scores = reciprocal_rank_fusion([bm25_ranked, vec_ranked], k=60, weights=(1.0, 1.0))
        candidates = [i for i, _ in sorted(rrf_scores.items(), key=lambda x: -x[1])[: max(15, k * 3)]]

        # 4. Cross-Encoder Reranking on the candidate pool (one batched model call)
        by_model = self.use_rerank and self.reranker.uses_model
        ce_scores = (self.reranker.score_batch(question, [(self.chunks[i].heading, self.chunks[i].text, self.chunks[i].clause_id)
                                                          for i in candidates], [self.model_texts[i] for i in candidates])
                     if self.use_rerank else [0.0] * len(candidates))
        hits: list[PolicyHit] = []
        for i, ce in zip(candidates, ce_scores):
            c = self.chunks[i]
            rrf = rrf_scores.get(i, 0.0)
            bm = bm25_scores.get(i, 0.0)
            if by_model:
                # A real cross-encoder read the question and the clause together: its judgement is the ranking.
                # BM25 and vectors only chose the candidates (standard two-stage retrieval).
                final_score = round(ce, 4)
            elif self.use_rerank:
                # Built-in scorer: combine it with RRF & lexical signal
                final_score = round(0.65 * ce + 0.25 * min(1.0, bm / 8.0) + 0.10 * min(1.0, rrf * 30.0), 4)
            else:
                final_score = round(0.5 * min(1.0, bm / 8.0) + 0.5 * min(1.0, rrf * 30.0), 4)
            hits.append(PolicyHit(
                doc=c.doc, title=c.title, doc_id=c.doc_id, clause_id=c.clause_id, heading=c.heading,
                text=c.text, score=final_score, version=c.version, status=c.status, trust=c.trust,
                access=c.access,
            ))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    # ------------------------------------------------------------------ answers
    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Split a markdown chunk into citable units: sentences, list items, table rows."""
        parts: list[str] = []
        buf: list[str] = []

        def flush():
            if buf:
                para = " ".join(buf)
                parts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
                buf.clear()

        lines = [l.strip() for l in text.splitlines()]
        for i, line in enumerate(lines):
            if not line or set(line) <= set("|-: "):
                flush()
                continue
            if line.startswith("|"):
                flush()
                is_header = i + 1 < len(lines) and lines[i + 1].startswith("|") and set(lines[i + 1]) <= set("|-: ")
                if not is_header:
                    parts.append(line)
            elif line.startswith(("- ", "* ")) or re.match(r"^\d+\.\s", line):
                flush()
                buf.append(line.lstrip("-* "))
            else:
                buf.append(line)
        flush()
        return parts

    def _extractive_answer(self, question: str, hits: list[PolicyHit]) -> str:
        q = set(tokenize(question))
        top = hits[0]
        sents = self._sentences(top.text)
        scored = sorted(((len(q & set(tokenize(s))), -i, s) for i, s in enumerate(sents)), reverse=True)
        best = [s for n, _, s in scored[:2] if n > 0] or (sents[:2] if sents else [top.text.strip()])
        label = top.clause_id or top.heading
        return f"{label}: " + " ".join(best) + f"  (source: {top.title}, v{top.version})"

    def answer(self, question: str, user_groups: Iterable[str] = ("public",), *, llm=None, k: int = 4,
               include_archived: bool = False) -> PolicyAnswer:
        hits = self.search(question, user_groups, k=k, include_archived=include_archived)
        if not (self.use_hybrid or self.use_rerank):
            threshold = self.BM25_ABSTAIN_BELOW
        else:
            threshold = self.MODEL_ABSTAIN_BELOW if (self.use_rerank and self.reranker.uses_model) else self.ABSTAIN_BELOW
        if not hits or hits[0].score < threshold:
            return PolicyAnswer(
                question=question, abstained=True, citations=hits[:2],
                answer="I could not find this in the policy documents you have access to. "
                       "If it concerns the Approved Vendor List, ask Procurement.",
            )

        # Data-egress rule: clauses restricted to Security / Procurement (e.g. the Approved Vendor List) are never
        # sent to a cloud model. They are answered by a model on this machine, or extractively with no model at all.
        restricted = [h for h in hits if not (set(h.access) & self.PUBLIC_GROUPS)]
        may_generate = llm is not None and not llm.mock_mode and (not restricted or llm.provider in LOCAL_PROVIDERS)
        if llm is not None and not llm.mock_mode and restricted and not may_generate:
            if llm.tracer:
                llm.tracer.event("egress_blocked", clauses=[h.clause_id or h.heading for h in restricted][:5],
                                 provider=llm.provider, action="answered extractively")
        if may_generate:
            sources_block = "\n\n".join(
                f"<source id={i+1} clause={h.clause_id or h.heading!r} title={h.title!r} version={h.version!r}>\n{h.text}\n</source>"
                for i, h in enumerate(hits)
            )
            prompt = f"Sources:\n{sources_block}\n\nQuestion: {question}"
            try:
                gen = llm.chat(POLICY_SYSTEM_PROMPT, prompt, stage="policy_generate")
                if "could not find this in the policy" in gen.lower():
                    return PolicyAnswer(question=question, answer=gen, citations=hits, abstained=True, mode="llm")
                # Ensure top clause ID and version are cited for traceability
                top = hits[0]
                label = top.clause_id or top.heading
                if label and label not in gen:
                    gen = f"{label}: {gen}"
                if f"v{top.version}" not in gen:
                    gen = f"{gen}  (source: {top.title}, v{top.version})"
                return PolicyAnswer(question=question, answer=gen, citations=hits, mode="llm")
            except Exception:  # noqa: BLE001 — fall back to extractive if live call fails
                pass

        text = self._extractive_answer(question, hits)
        return PolicyAnswer(question=question, answer=text, citations=hits, mode="extractive")

    # ------------------------------------------------------------------ the Approved Vendor List
    def approved_vendor_entry(self, vendor: str, user_groups: Iterable[str]) -> dict | str | None:
        """Look the vendor up in the restricted Approved Vendor List."""
        avl = [c for c in self.chunks if c.doc_id == "NM-PROC-AVL"]
        if not avl:
            return None
        if not any(self._allowed(c, user_groups) for c in avl):
            return "NO_ACCESS"
        v = vendor.lower()
        for c in avl:
            status = {"AVL-1": "approved", "AVL-2": "in_progress", "AVL-3": "rejected"}.get(c.clause_id, "")
            for line in c.text.splitlines():
                if not line.startswith("|") or set(line) <= set("|-: "):
                    continue
                cells = [x.strip() for x in line.strip("|").split("|")]
                if len(cells) < 2 or cells[0].lower() in ("vendor",):
                    continue
                if v in cells[0].lower() or v in cells[1].lower():
                    if status == "approved" and len(cells) >= 7:
                        tier = cells[2].split()[0] if cells[2] else ""
                        return {"status": status, "tier": tier, "plan": cells[3], "approved_on": cells[4],
                                "next_review": cells[5], "notes": cells[6]}
                    if status == "in_progress" and len(cells) >= 5:
                        return {"status": status, "requested_tier": cells[2].split()[0], "notes": cells[4]}
                    if status == "rejected" and len(cells) >= 4:
                        return {"status": status, "date": cells[2], "notes": cells[3]}
        return None
