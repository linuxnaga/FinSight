"""Exact financial numbers from SEC XBRL company facts (cached locally).

An 8B model cannot be trusted to read the right cell from a flattened table
or to know that Amazon reports `RevenueFromContractWithCustomerExcludingAssessedTax`
while American Express reports `RevenuesNetOfInterestExpense`. This tool takes a
plain-English metric name, tries the known tags in order, and returns the
value reported for that fiscal year in that year's own 10-K.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache

from finsight.config import COMPANIES, XBRL_DIR

# friendly metric -> candidate us-gaap tags, most common first
CONCEPTS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "RevenuesNetOfInterestExpense",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfSales"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "research_and_development": ["ResearchAndDevelopmentExpense",
                                 "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                                  "DepreciationAndAmortization"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capital_expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
                            "PaymentsForCapitalImprovements"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "share_repurchases": ["PaymentsForRepurchaseOfCommonStock"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "inventory": ["InventoryNet"],
    "accounts_receivable": ["AccountsReceivableNetCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "shareholders_equity": ["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
INSTANT = {"total_assets", "total_liabilities", "current_assets", "current_liabilities", "cash", "inventory",
           "accounts_receivable", "long_term_debt", "shareholders_equity"}

ALIASES = {
    "sales": "revenue", "net sales": "revenue", "total revenue": "revenue", "revenues": "revenue",
    "cogs": "cost_of_revenue", "cost of sales": "cost_of_revenue", "cost of goods sold": "cost_of_revenue",
    "operating profit": "operating_income", "ebit": "operating_income", "net earnings": "net_income",
    "profit": "net_income", "r&d": "research_and_development", "capex": "capital_expenditure",
    "capital expenditures": "capital_expenditure", "cfo": "operating_cash_flow",
    "cash from operations": "operating_cash_flow", "d&a": "depreciation_amortization", "eps": "eps_diluted",
    "buybacks": "share_repurchases", "equity": "shareholders_equity", "debt": "long_term_debt",
    "receivables": "accounts_receivable",
}


class XBRLError(ValueError):
    pass


@lru_cache(maxsize=32)
def _facts(ticker: str) -> dict:
    p = XBRL_DIR / f"{ticker}.json"
    if not p.exists():
        raise XBRLError(f"no XBRL data cached for {ticker}; run python -m finsight.data.ingest")
    return json.loads(p.read_text())


def normalize_metric(metric: str) -> str:
    m = metric.strip().lower().replace("-", " ")
    if m in CONCEPTS:
        return m
    m2 = m.replace(" ", "_")
    if m2 in CONCEPTS:
        return m2
    if m in ALIASES:
        return ALIASES[m]
    raise XBRLError(f"unknown metric '{metric}'. Known: {sorted(CONCEPTS)}")


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def get_fact(ticker: str, metric: str, fiscal_year: int) -> dict:
    ticker = ticker.upper()
    if ticker not in COMPANIES:
        raise XBRLError(f"{ticker} is not in the corpus. Covered: {sorted(COMPANIES)}")
    key = normalize_metric(metric)
    gaap = _facts(ticker).get("facts", {}).get("us-gaap", {})
    found = []
    for tag in CONCEPTS[key]:
        concept = gaap.get(tag)
        if not concept:
            continue
        for unit, rows in concept.get("units", {}).items():
            # the period ending in `fiscal_year`, as first reported (original 10-K, not later restatements)
            rows = [r for r in rows if r.get("form") == "10-K" and r["end"][:4] == str(fiscal_year)
                    and not r.get("segment")]
            if key in INSTANT:
                rows = [r for r in rows if "start" not in r]
            else:
                rows = [r for r in rows if "start" in r and 340 <= _days(r["start"], r["end"]) <= 380]
            if not rows:
                continue
            best = min(rows, key=lambda r: (r["filed"], -int(r["end"].replace("-", ""))))
            found.append({
                "ticker": ticker, "company": COMPANIES[ticker][1], "metric": key, "xbrl_tag": tag,
                "fiscal_year": fiscal_year, "value": best["val"], "unit": unit,
                "value_millions": round(best["val"] / 1e6, 2) if unit == "USD" else None,
                "period_start": best.get("start"), "period_end": best["end"], "accession": best["accn"],
                "source": f"SEC XBRL {ticker} FY{fiscal_year} 10-K us-gaap:{tag}",
            })
            if key != "revenue":
                return found[0]
    if found:
        # Revenue: filers use several tags, sometimes for sub-totals (AmEx reports
        # contract revenue AND total revenue net of interest). Total = the largest.
        return max(found, key=lambda f: f["value"])
    raise XBRLError(f"{ticker} has no '{key}' value for FY{fiscal_year} in XBRL (tried {CONCEPTS[key]}). "
                    f"Try search_filings instead.")
