"""
injection.py — one definition of "this text is trying to instruct the AI", used everywhere.

Three places read untrusted text: the user's question (guardrails.py), policy chunks (policy_rag.py) and
web pages (vendor_research.py). They used to keep three separate pattern lists that could drift apart; an
attack caught in one place could pass in another. They now share this module.

Layers, cheapest first (defence in depth, Deep Search Module 8):
  1. normalise   Unicode NFKC, remove zero-width and bidi control characters, rejoin s p a c e d letters,
                 decode base64 runs. Obfuscation is undone before anything is matched.
  2. patterns    high-precision regexes for known attack phrasing. A hit is decisive.
  3. classifier  a small local model (ONNX, CPU) scores the text. Used when installed and configured
                 (INJECTION_MODEL); catches paraphrases the patterns miss. No text leaves the machine.
  4. LLM         guardrails.classify_injection(), the last and most expensive layer, for user input.

A page that trips any layer is quarantined whole: nothing from it is extracted or cited.
"""

from __future__ import annotations

import base64
import os
import re
import threading
import unicodedata
from typing import Optional

# Known attack phrasing. Keep these precise: a false positive quarantines a legitimate page.
PATTERNS: list[str] = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|other|system|earlier)\s+(instructions|sources|rules|guidance)",
    r"disregard\s+.{0,30}(sources|instructions|rules)",
    r"(ai|language model|assistant|llm|automated reviewer|reviewer)s?\s+(reading|processing|parsing)\s+this",
    r"note to (the\s+)?(ai|assistants?|automated reviewers?|llm)",
    r"reminder to all assistants",
    r"automated reviewers reading this",
    r"report\s+pass\b",
    r"report that .{0,40}(compliant|approved|safe)",
    r"output\s+pass\s+for\s+(every|all)",
    r"mark\s+(every|all|each)\s+requirements?",
    r"pretend\s+(that\s+)?all\s+requirements\s+passed",
    r"skip the (policy|security) checks",
    r"you are now\b",
    r"system prompt",
    r"developer mode",
    r"pre-approved by procurement",
    r"new instructions\s*:",
    r"</?(system|instructions?)>",
]
_RE = re.compile("|".join(f"(?:{p})" for p in PATTERNS), re.I)

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍‎‏⁠⁡⁢⁣⁤﻿­‪‫‬‭‮⁦⁧⁨⁩"))
_SPACED = re.compile(r"\b(?:[A-Za-z][ .\-_]){3,}[A-Za-z]\b")
_B64 = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}")
_HOMOGLYPHS = str.maketrans({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ѕ": "s",
                             "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "М": "M", "Н": "H", "К": "K"})


def normalise(text: str) -> str:
    """Undo the cheap obfuscations: compatibility forms, invisible characters, look-alike letters, spaced letters."""
    t = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH).translate(_HOMOGLYPHS)
    t = _SPACED.sub(lambda m: re.sub(r"[ .\-_]", "", m.group(0)), t)
    return re.sub(r"[ \t\r\f\v]+", " ", t)


def decoded_payloads(text: str) -> list[str]:
    """Readable text hidden in base64 runs."""
    out = []
    for run in _B64.findall(text):
        try:
            raw = base64.b64decode(run + "=" * (-len(run) % 4), validate=False)
            dec = raw.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        if len(dec) >= 12 and sum(c.isprintable() for c in dec) / len(dec) > 0.9:
            out.append(dec)
    return out


def scan(text: str, hidden_text: str = "") -> list[str]:
    """Pattern layer. Returns the matched phrases (empty list = clean). `hidden_text` is content a browser
    would not show (HTML comments, display:none): instructions there are aimed at machines by definition."""
    norm = normalise(text)
    hits = [m.group(0) for m in _RE.finditer(norm)]
    for dec in decoded_payloads(text):
        hits += [f"base64:{m.group(0)}" for m in _RE.finditer(normalise(dec))]
    if hidden_text:
        hits += [f"hidden:{m.group(0)}" for m in _RE.finditer(normalise(hidden_text))]
        for dec in decoded_payloads(hidden_text):
            hits += [f"hidden-base64:{m.group(0)}" for m in _RE.finditer(normalise(dec))]
    return hits


def is_injection(text: str) -> bool:
    return bool(scan(text))


# ---------------------------------------------------------------------------
# Layer 3: optional local classifier (ONNX on CPU). Off unless INJECTION_MODEL is set and the files exist.
# ---------------------------------------------------------------------------

_CLF_LOCK = threading.Lock()
_CLF: object | None = None
_CLF_TRIED = False


class _OnnxClassifier:
    """A sequence-classification model exported to ONNX, with its tokenizer.json. Label names decide the
    'injection' column, so this works with any binary prompt-injection classifier that ships id2label."""

    def __init__(self, model_dir: str):
        import json
        from pathlib import Path

        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        onnx_files = sorted(d.glob("*.onnx")) or sorted(d.glob("onnx/*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx file under {d}")
        preferred = [f for f in onnx_files if f.name == "model.onnx"] or onnx_files
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("INJECTION_THREADS", "2"))
        self.session = ort.InferenceSession(str(preferred[0]), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").exists() else {}
        labels = {int(k): str(v).upper() for k, v in (cfg.get("id2label") or {}).items()}
        bad = [i for i, name in labels.items() if any(w in name for w in ("INJECTION", "JAILBREAK", "MALICIOUS", "UNSAFE", "LABEL_1"))]
        self.bad_index = bad[0] if bad else 1
        self.inputs = {i.name for i in self.session.get_inputs()}

    def score(self, text: str) -> float:
        import numpy as np

        # Long pages are scored in windows; one bad window is enough.
        words = text.split()
        windows = [" ".join(words[i:i + 300]) for i in range(0, max(1, len(words)), 250)][:8]
        worst = 0.0
        for w in windows:
            enc = self.tokenizer.encode(w)
            feed = {"input_ids": np.array([enc.ids], dtype=np.int64), "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
            logits = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][0]
            e = np.exp(logits - logits.max())
            worst = max(worst, float((e / e.sum())[self.bad_index]))
        return worst


def local_classifier():
    """The loaded classifier, or None when it is not configured / not installed."""
    global _CLF, _CLF_TRIED
    with _CLF_LOCK:
        if _CLF_TRIED:
            return _CLF
        _CLF_TRIED = True
        model_dir = os.environ.get("INJECTION_MODEL", "").strip()
        if not model_dir or os.environ.get("LLM_PROVIDER", "mock").strip().lower() == "mock":
            return None      # mock mode stays deterministic and dependency-free
        try:
            _CLF = _OnnxClassifier(model_dir)
        except Exception:  # noqa: BLE001 — missing files or packages: the other layers still run
            _CLF = None
        return _CLF


def classifier_score(text: str) -> Optional[float]:
    """Probability that `text` is an injection according to the local model, or None if no model is loaded."""
    clf = local_classifier()
    if clf is None or not text.strip():
        return None
    try:
        return clf.score(normalise(text))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None


def classifier_threshold() -> float:
    try:
        return float(os.environ.get("INJECTION_THRESHOLD", "0.99"))
    except ValueError:
        return 0.99
