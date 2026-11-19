"""Helpers that keep untrusted text out of notes, logs and traces (hard rule: nothing from a page or a model reply reaches them).

A note may carry a page id (only when it looks like one), a family or field name from the fixed vocabularies, and an exception
CLASS name. Anything else a model or a page wrote is replaced by a fixed phrase."""
from __future__ import annotations

import re

from .config import FAMILIES

_ECHO_ID = re.compile(r"[A-Za-z0-9_.:-]{1,12}")       # a real page id is short (e.g. c0060_p12); longer text could be a page phrase


def safe_id(page_id) -> str:
    """A page id fit for a note: a model-written id could carry page text, so only a short id-shaped value is echoed."""
    text = str(page_id)
    return text if _ECHO_ID.fullmatch(text) else f"<{len(text)} chars>"


def family_name(family) -> str:
    """A family name fit for a note: only a name from FAMILIES is echoed, whatever else a model wrote."""
    return family if isinstance(family, str) and family in FAMILIES else "an unknown family"
