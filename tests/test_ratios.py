import pytest

from finsight.agent.graph import detect_metrics
from finsight.config import XBRL_DIR, fiscal_year_of
from finsight.tools.ratios import detect_ratios
from eval.metrics import gold_decimals, numeric_match

has_data = pytest.mark.skipif(not (XBRL_DIR / "NKE.json").exists(), reason="run ingest first")


@pytest.mark.parametrize("end,fy", [
    ("2023-05-31", 2023),  # Nike
    ("2023-01-28", 2023),  # Best Buy names the year after the January it ends in
    ("2023-01-01", 2022),  # J&J 52/53-week year spilling into January
    ("2021-01-03", 2020),
    ("2022-12-31", 2022),
])
def test_fiscal_year_of(end, fy):
    assert fiscal_year_of(end) == fy


@pytest.mark.parametrize("q,expected", [
    ("What is Coca Cola's FY2021 COGS % margin?", ["cogs_margin"]),
    ("What is the FY2022 unadjusted EBITDA less capex for PepsiCo? Define unadjusted EBITDA as ...",
     ["ebitda_less_capex", "ebitda"]),
    ("General Mills's FY2020 working capital ratio?", ["current_ratio"]),
    ("Lockheed Martin's FY2021 net working capital?", ["working_capital"]),
    ("Roughly how many times has JnJ sold its inventory in FY2022?", ["inventory_turnover"]),
    ("What was Adobe's operating margin and ROE?", ["operating_margin", "return_on_equity"]),
])
def test_detect_ratios_in_order(q, expected):
    assert detect_ratios(q) == expected


@pytest.mark.parametrize("q,expected", [
    ("How much did AWK pay out in cash dividends? Use the statement of cash flows.", ["dividends_paid"]),
    ("What was KO's net cash provided by operating activities in 2023?", ["operating_cash_flow"]),
    ("What were Nike's cash and cash equivalents?", ["cash"]),
    ("What is PepsiCo's FY2021 capital expenditure amount?", ["capital_expenditure"]),
])
def test_detect_metrics_ignores_boilerplate(q, expected):
    assert detect_metrics(q) == expected


def test_numeric_match_handles_rounding_and_percent_vs_fraction():
    assert numeric_match(-1.531, "", -0.02, 0.01, gold_decimals("-0.02"))  # -1.53% vs "-0.02"
    assert numeric_match(4625.0, "", 4.6, 0.01, gold_decimals("$4.60"))    # millions vs billions
    assert not numeric_match(389.0, "", 0.4, 0.01, gold_decimals("$0.40"))


@has_data
def test_ratio_values():
    from finsight.tools.ratios import compute_ratio

    assert compute_ratio("NKE", "gross_margin", 2023)["value"] == pytest.approx(43.52, abs=0.01)
    assert compute_ratio("PFE", "growth", 2023, metric="revenue")["value"] == pytest.approx(-41.70, abs=0.01)
    r = compute_ratio("JNJ", "inventory_turnover", 2022)
    assert set(r["inputs"]) == {"cost_of_revenue FY2022", "inventory FY2022", "inventory FY2021"}


@has_data
def test_xbrl_original_filing_and_derived():
    from finsight.tools.xbrl import get_fact

    assert get_fact("JNJ", "research_and_development", 2021)["value_millions"] == 14714.0  # not the stray 0
    assert get_fact("JNJ", "eps_diluted", 2022)["value"] == 6.73
    assert get_fact("ADBE", "sga", 2022)["xbrl_tag"].startswith("derived")
