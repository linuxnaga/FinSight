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
    assert numeric_match(None, "capex was $1,749 million", -1749, 0.01)   # outflow stated as positive
    assert not numeric_match(389.0, "", 0.4, 0.01, gold_decimals("$0.40"))


@pytest.mark.parametrize("pred,text,gold,gold_text", [
    # real false positive from the first baseline run: any number scaled by 1e-6 rounded to 0.02
    (6.19, "AES's FY2022 ROA is 6.19% (Net income of $1,188 million divided by average total assets of "
           "($23,039 million + $24,557 million) / 2 = $23,803 million)", -0.02, "-0.02"),
    (None, "Revenue was $51,217 million in FY2023.", 0.0, "0"),       # gold 0 needs an explicit 0
    (None, "In FY2022 the ratio was 1.9.", 2022.0, "2022"),           # years are not answers
    (12.5, "Margin was 12.5%", 0.4, "0.4"),                           # wrong value, no rounding rescue
])
def test_numeric_match_rejects_false_positives(pred, text, gold, gold_text):
    assert not numeric_match(pred, text, gold, 0.01, gold_decimals(gold_text))


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


@pytest.mark.parametrize("gold,answer,expected", [
    # real judge false positives from the first Qwen baseline
    ("AES has converted inventory 9.5 times in FY 2022.",
     "AES's inventory turnover ratio for FY2022 is not meaningful as it has little inventory.", False),
    ("No, JnJ's FY2022 financials are not of a high growth company as sales grew by 1.3% in FY2022.",
     "JnJ's FY2022 financials do not indicate a high growth company. Revenue grew by 6% year-over-year.", False),
    ("The effective tax rate changed from 24.6% in FY 2021 to 21.6% in FY 2022.",
     "The effective tax rate was 22.6% for FY2021 and 21.8% for FY2022.", False),
    # genuinely right answers still pass
    ("The effective tax rate of Corning has changed from 20% in FY2021 to 23% in FY 2022.",
     "The effective tax rate was 21.6% for FY2021 and 23.1% for FY2022.", True),
    ("AES has converted inventory 9.5 times in FY 2022.", "Inventory turnover was about 9.46x.", True),
    # gold without figures -> gate abstains, the judge decides
    ("Yes, Boeing's business is subject to cyclicality.", "Boeing is cyclical.", None),
    ("Amcor operates in packaging across 40 countries in 2022.", "Packaging.", False),
])
def test_figure_gate(gold, answer, expected):
    from eval.metrics import figure_gate
    assert figure_gate(gold, answer) is expected


def test_judge_cache_and_figure_note(tmp_path, monkeypatch):
    import eval.metrics as M
    from tests.fakes import FakeClient
    from finsight.llm import LLM

    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    monkeypatch.setattr(M, "_cache_db", None)
    seen = []

    def respond(system, user, i):
        seen.append(user)
        return {"score": 5, "correct": True, "reason": "ok"}

    llm = LLM(client=FakeClient(respond), json_mode="none")
    q, gold = "Corning ETR change?", "The effective tax rate changed from 20% in FY2021 to 23% in FY2022."
    ans = "It increased from 20.48% to 22.87%."
    a = M.judge(llm, q, gold, ans)
    b = M.judge(llm, q, gold, ans)  # identical inputs: served from cache, no second model call
    assert a == b and len(seen) == 1
    assert "Automatic check" in seen[0] and "20, 23" in seen[0]
    # a cached "correct" verdict is still overruled when the answer states none of the gold figures
    assert M.judge(llm, q, gold, "It went up a bit.").correct is False
