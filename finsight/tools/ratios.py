"""Standard financial ratios computed deterministically from XBRL facts.

Asking an 8B model to plan "fetch A, fetch B, fetch last year's B, average,
divide" is where most multi-step failures happen. Here the model only has to
pick a ratio name; formulas, prior-year averages and units are code.

Formula variables are metric names from xbrl.CONCEPTS, with two modifiers:
  <metric>_prev   value for fiscal_year - 1
  avg_<metric>    average of fiscal_year and fiscal_year - 1 (balance-sheet items)
"""
from __future__ import annotations

import re

from finsight.tools import xbrl
from finsight.tools.calc import calculate

# name -> (formula, unit, definition shown to the user)
RATIOS: dict[str, tuple[str, str, str]] = {
    "gross_margin": ("gross_profit / revenue * 100", "%", "gross profit / revenue"),
    "operating_margin": ("operating_income / revenue * 100", "%", "operating income / revenue"),
    "net_margin": ("net_income / revenue * 100", "%", "net income / revenue"),
    "ebitda": ("operating_income + depreciation_amortization", "USD millions",
               "operating income + depreciation & amortization (unadjusted)"),
    "ebitda_margin": ("(operating_income + depreciation_amortization) / revenue * 100", "%",
                      "(operating income + D&A) / revenue"),
    "effective_tax_rate": ("income_tax / pretax_income * 100", "%", "income tax expense / pre-tax income"),
    "current_ratio": ("current_assets / current_liabilities", "x", "current assets / current liabilities"),
    "quick_ratio": ("(current_assets - inventory) / current_liabilities", "x",
                    "(current assets - inventory) / current liabilities"),
    "cash_ratio": ("cash / current_liabilities", "x", "cash and equivalents / current liabilities"),
    "debt_to_equity": ("long_term_debt / shareholders_equity", "x", "long-term debt / shareholders' equity"),
    "debt_to_assets": ("long_term_debt / total_assets", "x", "long-term debt / total assets"),
    "free_cash_flow": ("operating_cash_flow - capital_expenditure", "USD millions",
                       "cash from operations - capital expenditure"),
    "fcf_margin": ("(operating_cash_flow - capital_expenditure) / revenue * 100", "%", "free cash flow / revenue"),
    "fcf_conversion": ("(operating_cash_flow - capital_expenditure) / net_income * 100", "%",
                       "free cash flow / net income"),
    "capex_to_revenue": ("capital_expenditure / revenue * 100", "%", "capital expenditure / revenue"),
    "capex_to_depreciation": ("capital_expenditure / depreciation_amortization", "x", "capex / D&A"),
    "rnd_intensity": ("research_and_development / revenue * 100", "%", "R&D expense / revenue"),
    "sga_to_revenue": ("sga / revenue * 100", "%", "SG&A / revenue"),
    "interest_coverage": ("operating_income / interest_expense", "x", "operating income / interest expense"),
    "dividend_payout_ratio": ("dividends_paid / net_income * 100", "%", "dividends paid / net income"),
    "inventory_turnover": ("cost_of_revenue / avg_inventory", "x", "cost of revenue / average inventory"),
    "days_inventory_outstanding": ("365 / (cost_of_revenue / avg_inventory)", "days",
                                   "365 / inventory turnover"),
    "receivables_turnover": ("revenue / avg_accounts_receivable", "x", "revenue / average receivables"),
    "days_sales_outstanding": ("365 / (revenue / avg_accounts_receivable)", "days", "365 / receivables turnover"),
    "asset_turnover": ("revenue / avg_total_assets", "x", "revenue / average total assets"),
    "return_on_assets": ("net_income / avg_total_assets * 100", "%", "net income / average total assets"),
    "return_on_equity": ("net_income / avg_shareholders_equity * 100", "%", "net income / average equity"),
    "cogs_margin": ("cost_of_revenue / revenue * 100", "%", "cost of revenue / revenue"),
    "working_capital": ("current_assets - current_liabilities", "USD millions", "current assets - current liabilities"),
    "retention_ratio": ("(net_income - dividends_paid) / net_income", "x", "(net income - dividends paid) / net income"),
    "days_payable_outstanding": ("365 * avg_accounts_payable / cost_of_revenue", "days",
                                 "365 x average accounts payable / cost of revenue"),
    "ebitda_less_capex": ("operating_income + depreciation_amortization - capital_expenditure", "USD millions",
                          "operating income + D&A - capital expenditure"),
    "growth": ("({m} - {m}_prev) / abs({m}_prev) * 100", "%", "year-over-year change in {m}"),
}

# phrases that identify a ratio in a question (longest match wins)
RATIO_ALIASES: dict[str, str] = {
    "gross margin": "gross_margin", "gross profit margin": "gross_margin",
    "operating margin": "operating_margin", "operating profit margin": "operating_margin", "ebit margin": "operating_margin",
    "net margin": "net_margin", "net profit margin": "net_margin", "net income margin": "net_margin",
    "ebitda margin": "ebitda_margin", "ebitda % margin": "ebitda_margin", "ebitda less capex": "ebitda_less_capex",
    "ebitda": "ebitda", "operating income % margin": "operating_margin", "cogs % margin": "cogs_margin",
    "cogs margin": "cogs_margin", "cost of goods sold margin": "cogs_margin", "working capital ratio": "current_ratio",
    "net working capital": "working_capital", "working capital": "working_capital",
    "retention ratio": "retention_ratio", "days payable outstanding": "days_payable_outstanding",
    "dpo": "days_payable_outstanding", "gross profit % margin": "gross_margin", "net income % margin": "net_margin", "effective tax rate": "effective_tax_rate",
    "current ratio": "current_ratio", "quick ratio": "quick_ratio", "acid test": "quick_ratio",
    "cash ratio": "cash_ratio", "debt to equity": "debt_to_equity", "debt-to-equity": "debt_to_equity",
    "d/e ratio": "debt_to_equity", "debt to assets": "debt_to_assets", "free cash flow conversion": "fcf_conversion",
    "fcf conversion": "fcf_conversion", "free cash flow margin": "fcf_margin", "fcf margin": "fcf_margin",
    "free cash flow": "free_cash_flow", "fcf": "free_cash_flow", "capex to revenue": "capex_to_revenue",
    "capex as a % of revenue": "capex_to_revenue", "capex as a percentage of revenue": "capex_to_revenue",
    "capital intensity": "capex_to_revenue", "capital-intensive": "capex_to_revenue",
    "r&d intensity": "rnd_intensity", "r&d as a percentage of revenue": "rnd_intensity",
    "share of revenue was spent on r&d": "rnd_intensity", "revenue was spent on r&d": "rnd_intensity",
    "sg&a as a percentage of revenue": "sga_to_revenue", "interest coverage": "interest_coverage",
    "dividend payout": "dividend_payout_ratio", "payout ratio": "dividend_payout_ratio",
    "inventory turnover": "inventory_turnover", "sold its inventory": "inventory_turnover",
    "days inventory outstanding": "days_inventory_outstanding", "dio": "days_inventory_outstanding",
    "days sales outstanding": "days_sales_outstanding", "dso": "days_sales_outstanding",
    "receivables turnover": "receivables_turnover", "asset turnover": "asset_turnover",
    "return on assets": "return_on_assets", "roa": "return_on_assets",
    "return on equity": "return_on_equity", "roe": "return_on_equity",
}
GROWTH_RE = re.compile(r"\b(growth|grow|grew|change|changed|increase|increased|decrease|decreased|decline|declined|"
                       r"yoy|year[- ]over[- ]year)\b", re.I)


class RatioError(ValueError):
    pass


def detect_ratios(question: str) -> list[str]:
    """Ratios named in the question, in order of appearance (the first is the
    one being asked for; later mentions are usually part of its definition)."""
    q = re.sub(r"\bcash[\s-]?flows?\b", "cash flow", question.lower())
    hits: list[tuple[int, int, str]] = []
    for alias in sorted(RATIO_ALIASES, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w]){re.escape(alias)}(?![\w])", q):
            if not any(a <= m.start() < b for a, b, _ in hits):
                hits.append((m.start(), m.end(), RATIO_ALIASES[alias]))
    found: list[str] = []
    for _, _, r in sorted(hits):
        if r not in found:
            found.append(r)
    return found


def _value(f: dict) -> float:
    return f["value_millions"] if f.get("value_millions") is not None else f["value"]


def compute_ratio(ticker: str, ratio: str, fiscal_year: int, metric: str | None = None) -> dict:
    if ratio not in RATIOS:
        raise RatioError(f"unknown ratio '{ratio}'. Known: {sorted(RATIOS)}")
    formula, unit, definition = RATIOS[ratio]
    if ratio == "growth":
        if not metric:
            raise RatioError("ratio 'growth' needs metric=<one of the metrics>")
        m = xbrl.normalize_metric(metric)
        formula, definition = formula.format(m=m), definition.format(m=m.replace("_", " "))
        unit = "%"
    inputs: dict[str, dict] = {}
    variables: dict[str, float] = {}
    for name in sorted(set(re.findall(r"[a-z_]+", formula)) - {"abs"}):
        if name.startswith("avg_"):
            base = name[4:]
            cur, prev = xbrl.get_fact(ticker, base, fiscal_year), xbrl.get_fact(ticker, base, fiscal_year - 1)
            inputs[f"{base} FY{fiscal_year}"], inputs[f"{base} FY{fiscal_year - 1}"] = cur, prev
            variables[name] = (_value(cur) + _value(prev)) / 2
        elif name.endswith("_prev"):
            base = name[:-5]
            f = xbrl.get_fact(ticker, base, fiscal_year - 1)
            inputs[f"{base} FY{fiscal_year - 1}"] = f
            variables[name] = _value(f)
        else:
            f = xbrl.get_fact(ticker, name, fiscal_year)
            inputs[f"{name} FY{fiscal_year}"] = f
            variables[name] = _value(f)
    value = calculate(formula, variables)
    return {
        "ticker": ticker, "company": xbrl.COMPANIES[ticker][1], "ratio": ratio, "metric": metric,
        "fiscal_year": fiscal_year, "value": round(value, 4), "unit": unit, "definition": definition,
        "formula": formula, "inputs": {k: {"value": _value(f), "unit": "USD millions" if f.get("value_millions")
                                            is not None else f["unit"], "source": f["source"]}
                                       for k, f in inputs.items()},
        "source": f"SEC XBRL {ticker} FY{fiscal_year} 10-K; computed {ratio}",
    }
