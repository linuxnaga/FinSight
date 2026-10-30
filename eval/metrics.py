"""Scoring: numeric match, LLM-as-judge, refusal accuracy, citation hit,
retrieval recall@k, and the failure taxonomy."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Literal

from pydantic import BaseModel, Field

from finsight.agent.graph import numbers_in
from finsight.config import DATA_DIR

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


JUDGE_PROMPT = """You grade answers to financial questions against a gold answer. Be strict.
Rubric (score 1-5):
5 - same conclusion AND the same key figures as gold (rounding differences only)
4 - same conclusion, key figures match within 2%, a minor detail missing
3 - right direction, but a key figure is wrong or missing
2 - on topic, but the conclusion differs from gold or most figures are wrong
1 - contradicts gold, or says "not available / not meaningful / not provided" when gold has an answer
Rules: gold figures are often rounded. A candidate figure that rounds to the gold figure MATCHES
(20.48% matches gold "20%"; 22.87% matches "23%"; 0.540 matches "0.54").
A key figure that does not round to the gold figure caps the score at 3 even if the conclusion matches.
If the gold says "yes"/"consistent"/"increased" and the answer says the opposite, score 1 or 2.
Judge only correctness versus gold, not style or length."""

_FIG = re.compile(r"(?<![\w.])-?\$?\(?(\d[\d,]*\.?\d*)\)?\s*(%|percent|x\b|times\b)?", re.I)


def key_figures(gold_answer: str) -> list[tuple[float, float]]:
    """(value, tolerance) for each figure a correct answer must state:
    percentages, decimals and amounts in the gold answer; not years and not
    small counts ("3 segments"). The tolerance follows the gold's precision:
    "21.6%" must match to +-0.05 (half its last digit) and "23%" to +-0.5,
    and never tighter than 0.5% relative."""
    out = []
    for m in _FIG.finditer(gold_answer or ""):
        raw, unit = m.group(1).replace(",", ""), m.group(2)
        try:
            v = float(raw)
        except ValueError:
            continue
        is_int = "." not in raw
        if is_int and 1990 <= v <= 2035 and not unit:
            continue  # a year
        if is_int and v < 32 and not unit:
            continue  # a count or an ordinal
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append((v, max(0.5 * 10 ** -decimals, 0.005 * v)))
    return out


def matched_figures(gold_answer: str, answer: str) -> list[float] | None:
    """Gold key figures the answer states at the gold's precision (unit scales
    x1, x1e3, x1e-3). None if the gold states no key figures."""
    figs = key_figures(gold_answer)
    if not figs:
        return None
    cands = [n for n in numbers_in(answer or "") if not (float(n).is_integer() and 1990 <= n <= 2035)]
    return [g for g, tol in figs if any(abs(abs(c * sc) - g) <= tol for c in cands for sc in (1, 1e3, 1e-3))]


def figure_gate(gold_answer: str, answer: str) -> bool | None:
    """Deterministic check under the LLM judge. None if the gold states no key
    figures; otherwise True only if the answer states at least one of them at
    the gold's precision. The judge may fail an answer that passes, never
    rescue one that fails: a 7B judge accepted "turnover is not meaningful"
    for gold 9.5x."""
    m = matched_figures(gold_answer, answer)
    return None if m is None else bool(m)


_FIGURE_NOTE = ("Automatic check: the candidate states the gold's key figure(s) {figs} at the gold's precision "
                "(rounding already accounted for). Treat those figures as matching; grade the conclusion and "
                "completeness only.")
# verdicts are cached per rubric version: identical answers get identical verdicts across runs,
# so run-to-run differences measure the agent, not judge noise (a 7B judge flipped on identical text)
JUDGE_VERSION = hashlib.sha1((JUDGE_PROMPT + _FIGURE_NOTE).encode()).hexdigest()[:10]
_cache_db: sqlite3.Connection | None = None


def _cache() -> sqlite3.Connection:
    global _cache_db
    if _cache_db is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _cache_db = sqlite3.connect(DATA_DIR / "judge_cache.sqlite", check_same_thread=False)
        _cache_db.execute("CREATE TABLE IF NOT EXISTS j (k TEXT PRIMARY KEY, v TEXT)")
    return _cache_db


def judge(llm, question: str, gold: str, answer: str, use_cache: bool = True) -> Judgement:
    matched = matched_figures(gold, answer)
    user = f"Question: {question}\n\nGold answer: {gold}\n\nCandidate answer: {answer}"
    if matched:
        user += "\n\n" + _FIGURE_NOTE.format(figs=", ".join(f"{v:g}" for v in matched))
    key = hashlib.sha1(f"{JUDGE_VERSION}\x00{user}".encode()).hexdigest()
    row = _cache().execute("SELECT v FROM j WHERE k=?", (key,)).fetchone() if use_cache else None
    if row:
        j = Judgement.model_validate_json(row[0])
    else:
        j = llm.structured(JUDGE_PROMPT, user, Judgement, name="judge", max_tokens=150)
        j.correct = j.score >= 4
        if use_cache:
            _cache().execute("INSERT OR REPLACE INTO j VALUES (?, ?)", (key, j.model_dump_json()))
            _cache().commit()
    if j.correct and matched == []:  # the gold states figures and the answer states none of them
        j = j.model_copy(update={"correct": False, "reason": f"[figure gate: none of the gold's key figures "
                                 f"{[v for v, _ in key_figures(gold)]} stated] {j.reason}"})
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
