"""Tool interfaces the agent can call. Each tool has a Pydantic argument
schema (so malformed calls are rejected before execution), a timeout, and
returns either a result dict or {"error": ...} - tool failures never raise
into the agent loop.
"""
from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from finsight import tracing
from finsight.config import COMPANIES, FISCAL_YEARS
from finsight.tools import ratios, xbrl
from finsight.tools.calc import calculate

TickerT = Literal[tuple(COMPANIES)]  # type: ignore[valid-type]
MetricT = Literal[tuple(xbrl.CONCEPTS)]  # type: ignore[valid-type]
RatioT = Literal[tuple(ratios.RATIOS)]  # type: ignore[valid-type]


class SearchArgs(BaseModel):
    query: str = Field(description="what to look for, in filing language")
    ticker: TickerT | None = None
    fiscal_year: int | None = None
    item: str | None = Field(None, description="10-K item, e.g. '1A' risk factors, '7' MD&A, '8' financials")
    k: int = Field(5, ge=1, le=8)


class FactArgs(BaseModel):
    ticker: TickerT
    metric: MetricT
    fiscal_year: int


class CalcArgs(BaseModel):
    expression: str = Field(description="arithmetic using variable names, e.g. '(a-b)/b*100'")
    variables: dict[str, float | str] = Field(
        description="name -> number, or '$N' to use the value returned by step N")


class RatioArgs(BaseModel):
    ticker: TickerT
    ratio: RatioT
    fiscal_year: int
    metric: MetricT | None = Field(None, description="only for ratio='growth': which metric's YoY growth")


class PeerArgs(BaseModel):
    tickers: list[TickerT] = Field(min_length=2, max_length=6)
    metric: MetricT
    fiscal_years: list[int] = Field(min_length=1, max_length=4)


def search_years(fy: int | None) -> int | list[int] | None:
    """Years before the corpus start are still reported as comparatives in the
    next two 10-Ks (FY2019 figures appear in the FY2020 and FY2021 filings)."""
    if fy is None or fy in FISCAL_YEARS:
        return fy
    if FISCAL_YEARS[0] - 2 <= fy < FISCAL_YEARS[0]:
        return [y for y in (fy + 1, fy + 2) if y in FISCAL_YEARS]
    return fy


def _search(a: SearchArgs) -> dict:
    from finsight.retrieval.index import default_index

    hits = default_index().search(a.query, k=a.k, ticker=a.ticker, fiscal_year=search_years(a.fiscal_year),
                                  item=a.item)
    return {"passages": [{"id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "url": h.chunk.url,
                          "text": h.chunk.text[:1400]} for h in hits]}


def _fact(a: FactArgs) -> dict:
    return xbrl.get_fact(a.ticker, a.metric, a.fiscal_year)


def _calc(a: CalcArgs) -> dict:
    v = calculate(a.expression, {k: float(x) for k, x in a.variables.items()})
    return {"expression": a.expression, "variables": a.variables, "value": round(v, 6)}


def _ratio(a: RatioArgs) -> dict:
    return ratios.compute_ratio(a.ticker, a.ratio, a.fiscal_year, a.metric)


def _peers(a: PeerArgs) -> dict:
    rows, missing = [], []
    for t in a.tickers:
        for fy in a.fiscal_years:
            try:
                f = xbrl.get_fact(t, a.metric, fy)
                rows.append({"ticker": t, "fiscal_year": fy, "value": f["value"], "unit": f["unit"],
                             "source": f["source"]})
            except xbrl.XBRLError as e:
                missing.append(str(e))
    return {"metric": a.metric, "rows": rows, "missing": missing}


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[[Any], dict]
    timeout_s: float = 30.0


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("search_filings", "Search 10-K text (narrative, risks, MD&A, tables). Use filters whenever the "
         "company/year is known.", SearchArgs, _search),
    Tool("get_financial_fact", "Exact reported number for one company, metric and fiscal year from XBRL. "
         "Prefer this over search for any standard financial metric.", FactArgs, _fact, 10),
    Tool("compute_ratio", "A standard ratio for one company and fiscal year, computed from XBRL with a fixed "
         "formula (margins, liquidity, leverage, turnover, returns, free cash flow, YoY growth). Prefer this over "
         "fetching facts and calculating yourself.", RatioArgs, _ratio, 15),
    Tool("calculate", "Arithmetic on numbers from earlier steps (growth %, margins, ratios, differences). "
         "Never do maths yourself.", CalcArgs, _calc, 5),
    Tool("compare_peers", "Same metric for several companies/years in one call.", PeerArgs, _peers, 15),
]}

_pool = cf.ThreadPoolExecutor(max_workers=4)


def tool_catalog() -> str:
    lines = []
    for t in TOOLS.values():
        fields = ", ".join(f"{k}" + ("" if f.is_required() else "?") for k, f in t.args.model_fields.items())
        lines.append(f"- {t.name}({fields}): {t.description}")
    lines.append(f"  metrics: {', '.join(xbrl.CONCEPTS)}")
    lines.append(f"  ratios: {', '.join(ratios.RATIOS)}")
    lines.append(f"  tickers: {', '.join(f'{k}={v[1]}' for k, v in COMPANIES.items())}")
    lines.append(f"  fiscal years available: {FISCAL_YEARS}")
    return "\n".join(lines)


def validate_args(name: str, args: dict) -> BaseModel:
    if name not in TOOLS:
        raise ValueError(f"unknown tool '{name}'; available: {list(TOOLS)}")
    return TOOLS[name].args.model_validate(args)


def run_tool(name: str, args: dict) -> dict:
    """Validate, execute with timeout, and never raise."""
    with tracing.span(name, "tool", input=args) as sp:
        try:
            parsed = validate_args(name, args)
        except (ValueError, ValidationError) as e:
            sp.failure = "tool_call_error:bad_args"
            sp.output = {"error": f"invalid arguments: {str(e)[:400]}"}
            return sp.output
        fut = _pool.submit(TOOLS[name].fn, parsed)
        try:
            out = fut.result(timeout=TOOLS[name].timeout_s)
        except cf.TimeoutError:
            sp.failure = "tool_call_error:timeout"
            out = {"error": f"{name} timed out after {TOOLS[name].timeout_s}s"}
        except Exception as e:
            sp.failure = "tool_call_error:exception"
            out = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
        sp.output = out
        return out
