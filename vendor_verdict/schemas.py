"""Output contracts of the vendor_verdict agents. Each agent's JSON reply is validated against one of these models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, StrictInt


class FieldReading(BaseModel):
    """One statement a Field Specialist read on one page. `value` follows the rules_lite shape for `field`; for a list-valued
    field (certifications, subprocessors, incidents) it is one item object. `quote` is the verbatim page text it rests on."""
    page_id: str = ""         # empty for a single-page read; the caller forces the page's own id
    field: str
    value: Any
    quote: str = Field(min_length=1)
    display: str = ""        # the model's own short statement; the absence guard (claims.normalise_claim) reads it


class SpecialistOut(BaseModel):
    readings: list[FieldReading]


class SpecialistRaw(BaseModel):
    """The lenient reply shape the specialists are called with: each item is validated as a FieldReading on its own, so one
    malformed reading never costs the page its good ones."""
    readings: list[dict]


class PageRoute(BaseModel):
    page_id: str
    families: list[str]


class PlanOut(BaseModel):
    routes: list[PageRoute]


class RouterOut(BaseModel):
    families: list[str]


class GradeOut(BaseModel):
    grade: Literal["supported", "unsupported", "unsure"]


class DecisionOut(BaseModel):
    # Index into the numbered candidate list; None = abstain. Strict: no bool/str/float coercion, no negative index
    # (the caller checks the upper bound against its own candidate list).
    choice: StrictInt | None = Field(..., ge=0)
