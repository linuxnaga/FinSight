"""Deterministic pre-processing before any LLM call.

Small models are unreliable at (a) noticing they should refuse and (b)
extracting company/year consistently, so both are done with rules first:

  * refusal policy: investment advice, price prediction, personal data
  * PII masking on the user's input
  * entity extraction (tickers, fiscal years) with follow-up resolution
  * prompt-injection screening of retrieved passages (used by the executor)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from finsight.config import COMPANIES, FISCAL_YEARS

ALIASES: dict[str, str] = {
    "advanced micro devices": "AMD", "amd": "AMD", "boeing": "BA", "american express": "AXP", "amex": "AXP",
    "pepsico": "PEP", "pepsi": "PEP", "3m": "MMM", "adobe": "ADBE", "amcor": "AMCR", "best buy": "BBY",
    "bestbuy": "BBY", "verizon": "VZ", "corning": "GLW", "cvs": "CVS", "cvs health": "CVS",
    "general mills": "GIS", "nike": "NKE", "aes": "AES", "amazon": "AMZN", "american water works": "AWK",
    "american water": "AWK", "coca-cola": "KO", "coca cola": "KO", "coke": "KO", "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ", "j&j": "JNJ", "jnj": "JNJ", "lockheed martin": "LMT", "lockheed": "LMT", "pfizer": "PFE",
}
# Companies a user might ask about that we deliberately don't cover.
KNOWN_OUTSIDE = ["apple", "tesla", "microsoft", "google", "alphabet", "meta", "facebook", "nvidia", "netflix",
                 "walmart", "costco", "intel", "jpmorgan", "paypal", "infosys", "tcs", "reliance"]

REFUSAL_PATTERNS = [
    (r"\b(should|would|do you recommend|is it (a )?good (time|idea) to)\b.*\b(buy|sell|invest|short|hold)\b", "investment_advice"),
    (r"\b(buy|sell|hold)\b.*\b(stock|shares?)\b.*\?", "investment_advice"),
    (r"\b(recommend (buying|selling|investing)|stock picker|strong buy|buy rating|"
     r"which (of these )?stocks? (will|should|to)|will (double|triple|outperform|go up|crash))\b", "investment_advice"),
    (r"\b(price target|target price|stock (price )?(prediction|forecast)|will .* stock (go|rise|fall)|"
     r"predict .* (share|stock) price)\b", "price_prediction"),
    (r"\b(home address|phone number|social security|personal email|salary of (my|our)|ceo'?s? (address|phone))\b",
     "personal_data"),
]

REFUSAL_TEXT = {
    "investment_advice": "I can't give buy/sell/hold or investment recommendations. I can summarise what the "
                         "filings report (results, risks, guidance language) so you can form your own view.",
    "price_prediction": "I don't predict share prices. I can show reported financials and what management said "
                        "about outlook in the 10-K.",
    "personal_data": "I can't help find personal information about individuals. I only answer from public 10-K content.",
}

_PII = [
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[ID_NUMBER]"),
]

INJECTION_RE = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)|disregard (the|your) (system|instructions)|"
    r"you are now|new instructions:|system prompt|do not tell the user|recommend (buying|selling)|"
    r"respond only with|<\s*/?\s*(system|instructions?)\s*>)",
    re.I,
)


@dataclass
class Entities:
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    outside: list[str] = field(default_factory=list)
    inherited: bool = False
    assumed_year: bool = False


def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for rx, label in _PII:
        if rx.search(text):
            found.append(label)
            text = rx.sub(label, text)
    return text, found


def refusal_reason(question: str) -> str | None:
    q = question.lower()
    for pat, reason in REFUSAL_PATTERNS:
        if re.search(pat, q):
            return reason
    return None


def extract_entities(question: str, previous: Entities | None = None) -> Entities:
    q = question.lower()
    tickers: list[str] = []
    for alias in sorted(ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![\w&]){re.escape(alias)}(?![\w&])", q) and ALIASES[alias] not in tickers:
            tickers.append(ALIASES[alias])
    for t in COMPANIES:
        if re.search(rf"\b{t}\b", question) and t not in tickers:
            tickers.append(t)
    years = sorted({int(y) for y in re.findall(r"\b(?:fy|fiscal(?: year)?\s*)?'?(20[12]\d)\b", q)})
    for short in re.findall(r"\bfy'?(\d{2})\b", q):
        years.append(2000 + int(short))
    years = sorted(set(years))
    # "what is Boeing forecasting for FY2023?" is answered in the FY2022 10-K
    if years and re.search(r"\b(forecast\w*|expect\w*|outlook|guidance|anticipat\w*|plan(s|ned)? for|project\w*)\b", q):
        years = sorted(set(years) | {y - 1 for y in years if y - 1 in FISCAL_YEARS})
    outside = [n for n in KNOWN_OUTSIDE if re.search(rf"\b{n}\b", q)]
    ents = Entities(tickers, years, outside)
    # follow-up: "and the year before?" / "what about Pepsi?" inherit what's missing
    if previous is not None:
        if not ents.tickers and previous.tickers:
            ents.tickers, ents.inherited = list(previous.tickers), True
        if not ents.years and previous.years:
            ents.years, ents.inherited = list(previous.years), True
        if re.search(r"\b(year before|previous year|prior year|last year)\b", q) and previous.years and \
                not re.findall(r"20[12]\d", q):
            ents.years, ents.inherited = [min(previous.years) - 1], True
    return ents


def out_of_range_years(ents: Entities) -> list[int]:
    """Years we cannot answer. The two years before the corpus start are still
    covered: they appear as comparatives in the FY2020/FY2021 10-Ks."""
    return [y for y in ents.years if not (FISCAL_YEARS[0] - 2 <= y <= FISCAL_YEARS[-1])]


def screen_passage(text: str) -> tuple[str, bool]:
    """Neutralise instruction-like text inside retrieved documents. The
    passage is kept (it may still hold facts) but the suspicious span is
    replaced so the generator never sees it as an instruction."""
    if INJECTION_RE.search(text):
        return INJECTION_RE.sub("[removed: instruction-like text in source document]", text), True
    return text, False
