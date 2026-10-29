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

from finsight.config import COMPANIES, XBRL_DIR, fiscal_year_of

# friendly metric -> candidate us-gaap tags, most common first
CONCEPTS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "RevenuesNetOfInterestExpense",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfSales"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "research_and_development": ["ResearchAndDevelopmentExpense",
                                 "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
                                 "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestAndDebtExpense",
                         "InterestExpenseDebt"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                                  "DepreciationAndAmortization"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capital_expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
                            "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherProductiveAssets"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock", "PaymentsOfOrdinaryDividends"],
    "share_repurchases": ["PaymentsForRepurchaseOfCommonStock"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "inventory": ["InventoryNet", "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings",
                  "InventoryFinishedGoodsNetOfReserves"],
    "accounts_payable": ["AccountsPayableCurrent", "AccountsPayableTradeCurrent",
                         "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "accounts_receivable": ["AccountsReceivableNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent",
                            "ReceivablesNetCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"],
    "shareholders_equity": ["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    # building blocks for derived values (not usually asked for directly)
    "liabilities_and_equity": ["LiabilitiesAndStockholdersEquity"],
    "total_equity": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
    "selling_marketing": ["SellingAndMarketingExpense"],
    "general_admin": ["GeneralAndAdministrativeExpense"],
}

# Used only when no reported tag exists. Results are labelled "derived".
DERIVED: dict[str, tuple[str, list[str]]] = {
    "gross_profit": ("revenue - cost_of_revenue", ["revenue", "cost_of_revenue"]),
    "total_liabilities": ("liabilities_and_equity - total_equity", ["liabilities_and_equity", "total_equity"]),
    "sga": ("selling_marketing + general_admin", ["selling_marketing", "general_admin"]),
}
INSTANT = {"total_assets", "total_liabilities", "current_assets", "current_liabilities", "cash", "inventory",
           "accounts_receivable", "accounts_payable", "long_term_debt", "shareholders_equity", "liabilities_and_equity",
           "total_equity"}

ALIASES = {
    "sales": "revenue", "net sales": "revenue", "total revenue": "revenue", "revenues": "revenue",
    "cogs": "cost_of_revenue", "cost of sales": "cost_of_revenue", "cost of goods sold": "cost_of_revenue",
    "operating profit": "operating_income", "ebit": "operating_income", "net earnings": "net_income",
    "profit": "net_income", "r&d": "research_and_development", "capex": "capital_expenditure",
    "capital expenditures": "capital_expenditure", "cfo": "operating_cash_flow",
    "cash from operations": "operating_cash_flow", "d&a": "depreciation_amortization", "eps": "eps_diluted",
    "buybacks": "share_repurchases", "equity": "shareholders_equity", "debt": "long_term_debt",
    "receivables": "accounts_receivable", "net ar": "accounts_receivable", "accounts receivable": "accounts_receivable",
    "accounts payable": "accounts_payable", "payables": "accounts_payable",
    "capital expenditure": "capital_expenditure", "capital spending": "capital_expenditure",
    "net cash provided by operating activities": "operating_cash_flow", "cash flow from operations": "operating_cash_flow",
    "operating cash flows": "operating_cash_flow", "cash dividends": "dividends_paid", "dividends": "dividends_paid",
    "net income attributable": "net_income", "total revenues": "revenue", "net revenues": "revenue",
    "research and development": "research_and_development", "diluted eps": "eps_diluted",
    "diluted earnings per share": "eps_diluted", "basic eps": "eps_basic", "total debt": "long_term_debt",
    "pre-tax income": "pretax_income", "income before income taxes": "pretax_income",
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


def _reported(ticker: str, key: str, fiscal_year: int) -> dict | None:
    gaap = _facts(ticker).get("facts", {}).get("us-gaap", {})
    found = []
    for tag in CONCEPTS[key]:
        concept = gaap.get(tag)
        if not concept:
            continue
        for unit, rows in concept.get("units", {}).items():
            # the period ending in `fiscal_year`, as first reported (original 10-K, not later restatements)
            rows = [r for r in rows if r.get("form") == "10-K" and fiscal_year_of(r["end"]) == fiscal_year
                    and not r.get("segment")]
            if key in INSTANT:
                rows = [r for r in rows if "start" not in r]
            else:
                rows = [r for r in rows if "start" in r and 340 <= _days(r["start"], r["end"]) <= 380]
            if not rows:
                continue
            first = min(rows, key=lambda r: (r["filed"], -int(r["end"].replace("-", ""))))
            # one filing can repeat a period with stray values (J&J tags a 0 R&D next to the
            # real figure): within the original filing, take the largest magnitude
            same = [r for r in rows if r["accn"] == first["accn"] and r["end"] == first["end"]]
            best = max(same, key=lambda r: abs(r["val"]))
            found.append({
                "ticker": ticker, "company": COMPANIES[ticker][1], "metric": key, "xbrl_tag": tag,
                "fiscal_year": fiscal_year, "value": best["val"], "unit": unit,
                "value_millions": round(best["val"] / 1e6, 2) if unit == "USD" else None,
                "period_start": best.get("start"), "period_end": best["end"], "accession": best["accn"],
                "source": f"SEC XBRL {ticker} FY{fiscal_year} 10-K us-gaap:{tag}",
                "_filed": best["filed"], "_order": len(found),
            })
    if not found:
        return None
    if key == "revenue":
        # filers use several revenue tags, sometimes for sub-totals (AmEx reports
        # contract revenue AND total revenue net of interest). Total = the largest.
        pick = max(found, key=lambda f: f["value"])
    else:
        # prefer the figure from the original 10-K; a tag that first appears in a
        # later filing is a restatement (or, for J&J R&D, a stray 0)
        pick = min(found, key=lambda f: (f["_filed"], f["_order"]))
    return {k: v for k, v in pick.items() if not k.startswith("_")}


def get_fact(ticker: str, metric: str, fiscal_year: int) -> dict:
    ticker = ticker.upper()
    if ticker not in COMPANIES:
        raise XBRLError(f"{ticker} is not in the corpus. Covered: {sorted(COMPANIES)}")
    key = normalize_metric(metric)
    fact = _reported(ticker, key, fiscal_year)
    if fact is not None:
        return fact
    if key in DERIVED:
        formula, parts = DERIVED[key]
        got = {p: _reported(ticker, p, fiscal_year) for p in parts}
        if all(got.values()):
            from finsight.tools.calc import calculate

            val = calculate(formula, {p: f["value"] for p, f in got.items()})
            first = next(iter(got.values()))
            return {**first, "metric": key, "xbrl_tag": f"derived: {formula}", "value": val,
                    "value_millions": round(val / 1e6, 2),
                    "source": f"SEC XBRL {ticker} FY{fiscal_year} 10-K derived {key} = " +
                              " ; ".join(f"{p} us-gaap:{f['xbrl_tag']}" for p, f in got.items())}
    raise XBRLError(f"{ticker} has no '{key}' value for FY{fiscal_year} in XBRL (tried {CONCEPTS[key]}). "
                    f"Try search_filings instead.")
