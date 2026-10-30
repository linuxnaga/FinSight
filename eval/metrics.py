"""Scoring: numeric match, LLM-as-judge, refusal accuracy, citation hit,
retrieval recall@k, and the failure taxonomy."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from finsight.agent.graph import numbers_in

FAILURE_TYPES = ["retrieval_miss", "reasoning_error", "tool_call_error", "hallucination", "refusal_error"]


def gold_decimals(gold_answer: str) -> int | None:
    """Decimal places the gold answer was rounded to ("-0.02" -> 2), if it is a bare number."""
    m = re.fullmatch(r"\s*-?\$?\s*-?[\d,]*\.(\d+)\s*%?\s*", gold_answer or "")
    return len(m.group(1)) if m else None


# millions <-> billions/thousands, and percent <-> fraction ("ROA -1.53%" vs gold "-0.02")
_SCALES = (1, 1e3, 1e-3, 1e2, 1e-2)


def numeric_match(pred_value: float | None, answer_text: str, gold: float, tol: float = 0.01,
                  decimals: int | None = None) -> bool:
    """True if the answer states the gold number.

    * the headline value or any number in the text within `tol` (relative), at
      a unit scale from _SCALES; the sign may differ only here, for outflows
      stated as positives ("capex of 1,749" vs gold -1,749)
    * gold rounded coarsely ("-0.02"): only the HEADLINE value may match by
      rounding to the gold's decimals, same sign, scale 1 or percent<->fraction.
      Applying that to every number in the text accepted nearly anything.
    * gold 0: only an exact 0 counts
    """
    # years and small integers in prose ("/ 2", "3 segments") are not answers
    texts = [n for n in numbers_in(answer_text)
             if not (float(n).is_integer() and (1990 <= n <= 2035 or abs(n) < 32))]
    cands = ([pred_value] if pred_value is not None else []) + texts
    if gold == 0:
        return any(c == 0 for c in cands)
    for c in cands:
        for scale in _SCALES:
            v = c * scale
            if min(abs(v - gold), abs(v + gold)) / abs(gold) <= tol:
                return True
    if pred_value is not None and decimals is not None:
        for scale in (1, 1e-2, 1e2):
            if round(pred_value * scale, decimals) == round(gold, decimals) and \
                    (pred_value == 0 or (pred_value > 0) == (gold > 0)):
                return True
    return False


REFUSAL_MARKERS = re.compile(r"\b(can't|cannot|can not|don't|do not|not able|unable|isn't|is not|outside)\b.*"
                             r"\b(advice|recommend|predict|personal|corpus|cover|filings i cover|in my corpus)\b",
                             re.I | re.S)
NOT_FOUND_MARKERS = re.compile(r"\b(not (found|available|disclosed|reported|in the evidence|mentioned)|"
                               r"no (information|evidence|data)|does not (contain|mention|report|disclose)|"
                               r"insufficient|could not find|unable to find)\b", re.I)


def refused(result: dict) -> bool:
    return bool(result.get("refused")) or bool(REFUSAL_MARKERS.search(result.get("answer", "")))


def said_not_found(result: dict) -> bool:
    return bool(result.get("insufficient_evidence")) or bool(NOT_FOUND_MARKERS.search(result.get("answer", "")))


def citation_hit(result: dict, gold_ids: list[str]) -> bool | None:
    if not gold_ids:
        return None
    cited = {c["source"] for c in result.get("citations", [])}
    cited |= {re.sub(r"#\d+$", "", c) for c in cited}
    # XBRL facts are authoritative evidence for the financial-statement item
    if any(c.startswith("SEC XBRL") for c in cited):
        return True
    return bool(cited & set(gold_ids))


def recall_at_k(retrieved_doc_ids: list[str], gold_ids: list[str], k: int) -> float | None:
    if not gold_ids:
        return None
    top = set(retrieved_doc_ids[:k])
    return sum(g in top for g in gold_ids) / len(gold_ids)


# ------------------------------------------------------------ LLM judge
class Judgement(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = fully correct and complete, 1 = wrong")
    correct: bool = Field(description="true if score >= 4")
    reason: str = Field(description="one sentence")


JUDGE_PROMPT = """You grade answers to financial questions against a gold answer.
Rubric (score 1-5):
5 - same conclusion and key figures as gold (small rounding differences are fine)
4 - correct conclusion, minor detail missing or slightly different figure (<2%)
3 - partially correct: right direction but a key figure or part is wrong/missing
2 - mostly wrong but on topic
1 - wrong, contradicts gold, or refuses/says not found when gold has an answer
Judge only correctness versus gold, not style."""


def judge(llm, question: str, gold: str, answer: str) -> Judgement:
    user = f"Question: {question}\n\nGold answer: {gold}\n\nCandidate answer: {answer}"
    j = llm.structured(JUDGE_PROMPT, user, Judgement, name="judge", max_tokens=150)
    j.correct = j.score >= 4
    return j


# ---------------------------------------------------------- taxonomy
def classify_failure(item: dict, result: dict, retrieved_doc_ids: list[str]) -> str | None:
    """Assign one failure type to an incorrect result, in priority order.
    Uses the trace-derived fields the agent returns."""
    if item["should_refuse"] != refused(result):
        return "refusal_error"
    fails = " ".join(result.get("failures", []))
    if "tool_call_error" in fails or "budget_exceeded" in fails or "step_cap" in fails:
        return "tool_call_error"
    used_fact = any(c["source"].startswith("SEC XBRL") for c in result.get("citations", []))
    gold_ids = item.get("evidence_doc_ids") or []
    if gold_ids and not used_fact and not (set(gold_ids) & set(retrieved_doc_ids)):
        return "retrieval_miss"
    if any("numbers not found in evidence" in " ".join(c.get("issues", [])) for c in result.get("critiques", [])) \
            and result.get("critiques", [{}])[-1].get("action") != "accept":
        return "hallucination"
    return "reasoning_error"
