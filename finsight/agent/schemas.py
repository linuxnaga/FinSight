from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from finsight.tools.registry import TOOLS

ToolName = Literal[tuple(TOOLS)]  # type: ignore[valid-type]
RouteName = Literal["lookup", "calculation", "comparison", "narrative", "out_of_scope"]


class Route(BaseModel):
    route: RouteName
    metrics: list[str] = Field(default_factory=list, description="financial metrics mentioned, if any")


class PlanStep(BaseModel):
    tool: ToolName
    args: dict[str, Any]
    purpose: str = ""


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=6)


class Answer(BaseModel):
    answer: str = Field(description="2-6 sentences; cite evidence like [E1]")
    value: float | None = Field(None, description="the single headline number, if the question asks for one")
    unit: str | None = Field(None, description="e.g. 'USD millions', '%', 'USD per share'")
    citations: list[str] = Field(default_factory=list, description="evidence ids used, e.g. ['E1','E3']")
    insufficient_evidence: bool = False


class Critique(BaseModel):
    supported: bool = Field(description="every claim is backed by the evidence")
    answers_question: bool
    issues: list[str] = Field(default_factory=list)
    action: Literal["accept", "rewrite", "replan"] = "accept"
