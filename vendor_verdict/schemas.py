"""Output contracts of the vendor_verdict agents. Each agent's JSON reply is validated against one of these models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class FieldReading(BaseModel):
    """One statement a Field Specialist read on one page. `value` follows the rules_lite shape for `field`; for a list-valued
    field (certifications, subprocessors, incidents) it is one item object. `quote` is the verbatim page text it rests on."""
    page_id: str
    field: str
    value: Any
    quote: str


class SpecialistOut(BaseModel):
    readings: list[FieldReading]


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
    choice: int | None        # index into the numbered candidate list; None = abstain
