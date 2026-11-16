"""models.py — contracts between the detectors, the diagnose agent, the CLI and the fix agent."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from assessor.schemas import BaseModel

IssueKind = Literal["test", "runtime", "eval", "manual"]
FixType = Literal["code", "config", "needs_info", "no_fix"]


class Issue(BaseModel):
    id: str
    kind: IssueKind
    title: str
    signature: str
    evidence: str = ""                       # traceback / output, already secret-masked and truncated
    related: list[str] = Field(default_factory=list)   # test node ids or file paths
    flaky: bool = False
    note: str = ""                           # e.g. "previously fixed in <commit>"


class FixProposal(BaseModel):
    fix_type: FixType = "code"
    root_cause: str
    evidence: list[str] = Field(default_factory=list)          # file:line references
    plan: list[str] = Field(default_factory=list)
    files_to_edit: list[str] = Field(default_factory=list)     # existing files only
    new_files: list[str] = Field(default_factory=list)         # files the fix will create
    verification_tests: list[str] = Field(default_factory=list)
    risk: str = ""
    confidence: float = 0.5
    questions: list[str] = Field(default_factory=list)         # for fix_type == needs_info
    config_instructions: str = ""                              # for fix_type == config


class LoopResult(BaseModel):
    reason: str                                # done | max_steps | loop | error:<msg>
    final_text: str = ""
    steps: int = 0
    tool_calls: list[str] = Field(default_factory=list)


class FixResult(BaseModel):
    status: Literal["fixed", "not_fixed", "stopped"]
    loop_reason: str = ""
    summary: str = ""
    edited: list[str] = Field(default_factory=list)
    created: list[str] = Field(default_factory=list)
    targeted_ok: Optional[bool] = None
    new_failures: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)             # cheating-diff warnings
