"""End-to-end agent loop with a scripted model and the real tools/data."""
import json

import pytest

from finsight.agent.graph import Agent
from finsight.agent.memory import SessionStore
from finsight.config import XBRL_DIR
from finsight.llm import LLM
from tests.fakes import FakeClient, by_role

pytestmark = pytest.mark.skipif(not (XBRL_DIR / "NKE.json").exists(), reason="run ingest first")


def make_agent(responder, tmp_path):
    return Agent(llm=LLM(client=FakeClient(responder), json_mode="none"),
                 store=SessionStore(tmp_path / "s.sqlite"))


def test_lookup_fast_path_skips_planner(tmp_path):
    def answer(user):
        assert "51,217" in user
        return {"answer": "Nike's FY2023 revenue was $51,217 million [E1].", "value": 51217,
                "unit": "USD millions", "citations": ["E1"]}

    agent = make_agent(by_role(router={"route": "lookup"}, answer=answer), tmp_path)
    r = agent.run("What was Nike's revenue in FY2023?")
    assert r["value"] == 51217 and r["tool_calls"] == 1 and r["citations"][0]["id"] == "E1"
    assert r["plan"][0]["purpose"] == "direct lookup"


def test_growth_uses_ratio_fast_path(tmp_path):
    def answer(user):
        assert "revenue YoY growth = 9.65" in user
        return {"answer": "Nike revenue grew 9.65% from FY2022 to FY2023 [E1].", "citations": ["E1"]}

    agent = make_agent(by_role(answer=answer), tmp_path)  # no router/planner scripted: must not be called
    r = agent.run("What was Nike's revenue growth from FY2022 to FY2023?")
    assert r["route"] == "calculation" and r["plan"][0]["tool"] == "compute_ratio"
    assert r["value"] == pytest.approx(9.65, abs=0.01)
    assert r["critiques"][0]["action"] == "accept"


def test_ratio_with_driver_question_adds_search(tmp_path, monkeypatch):
    from finsight.tools import registry
    monkeypatch.setitem(registry.TOOLS, "search_filings", registry.Tool(
        "search_filings", "", registry.SearchArgs,
        lambda a: {"passages": [{"id": "x#0", "doc_id": "NKE_2023_10K_item7", "url": None,
                                 "text": "Gross margin decreased due to higher product costs and markdowns."}]}))
    agent = make_agent(by_role(answer={"answer": "Nike gross margin was 43.52% [E1], down on higher costs [E2].",
                                       "citations": ["E1", "E2"]}), tmp_path)
    r = agent.run("What drove Nike's gross margin in FY2023?")
    assert [p["tool"] for p in r["plan"]] == ["compute_ratio", "search_filings"]


def test_planner_step_references(tmp_path):
    plan = {"steps": [
        {"tool": "get_financial_fact", "args": {"ticker": "NKE", "metric": "revenue", "fiscal_year": 2023}},
        {"tool": "get_financial_fact", "args": {"ticker": "NKE", "metric": "shares_diluted", "fiscal_year": 2023}},
        {"tool": "calculate", "args": {"expression": "a/b", "variables": {"a": "$1", "b": "$2"}}},
    ]}
    agent = make_agent(by_role(router={"route": "calculation"}, planner=plan,
                               answer=lambda u: {"answer": "Nike revenue per diluted share [E3].", "citations": ["E3"]}),
                       tmp_path)
    r = agent.run("Compute Nike's FY2023 revenue per diluted share")
    calc = [o for o in r["plan"] if o["tool"] == "calculate"]
    assert calc and r["value"] is not None and r["value"] > 0


def test_no_year_defaults_to_latest(tmp_path):
    agent = make_agent(by_role(answer={"answer": "Nike revenue $51,217 million [E1].", "citations": ["E1"]}), tmp_path)
    r = agent.run("What is Nike's revenue?")
    assert r["answer"].startswith("(No fiscal year given; using FY2023")


def test_critic_catches_hallucinated_number_and_rewrites(tmp_path):
    drafts = iter([
        {"answer": "Nike's FY2023 revenue was $55,000 million [E1].", "value": 55000, "citations": ["E1"]},
        {"answer": "Nike's FY2023 revenue was $51,217 million [E1].", "value": 51217, "citations": ["E1"]},
    ])
    agent = make_agent(by_role(router={"route": "lookup"}, answer=lambda u: next(drafts)), tmp_path)
    r = agent.run("What was Nike's revenue in FY2023?")
    assert r["critiques"][0]["action"] == "rewrite"
    assert "55000" in json.dumps(r["critiques"][0]["issues"]) or "55,000" in json.dumps(r["critiques"][0]["issues"])
    assert r["value"] == 51217


def test_bad_planner_args_are_repaired(tmp_path):
    plans = iter([
        {"steps": [{"tool": "get_financial_fact", "args": {"ticker": "NIKE", "metric": "sales", "fiscal_year": 2023}}]},
        {"steps": [{"tool": "get_financial_fact", "args": {"ticker": "NKE", "metric": "gross_profit", "fiscal_year": 2023}}]},
    ])
    agent = make_agent(by_role(router={"route": "narrative"}, planner=lambda u: next(plans),
                               answer={"answer": "Nike gross profit was $22,292 million [E1].", "citations": ["E1"]},
                               critic={"supported": True, "answers_question": True}), tmp_path)
    r = agent.run("Tell me about Nike gross profit and what drove it in 2023")
    assert "tool_call_error:bad_args" in r["failures"]
    assert r["plan"][0]["args"]["ticker"] == "NKE"


def test_refusal_and_out_of_corpus_use_no_llm(tmp_path):
    agent = make_agent(lambda s, u, i: pytest.fail("LLM must not be called"), tmp_path)
    assert agent.run("Should I buy Boeing stock?")["refused"] == "investment_advice"
    assert agent.run("What was Tesla's revenue in 2022?")["refused"] == "out_of_corpus"
    assert agent.run("What was Nike's revenue in 2015?")["refused"] == "out_of_corpus"


def test_followup_inherits_company_and_year(tmp_path):
    agent = make_agent(by_role(router={"route": "lookup"},
                               answer=lambda u: {"answer": "See [E1]. Nike", "citations": ["E1"]}), tmp_path)
    agent.run("What was Nike's net income in FY2023?", session="s1")
    r = agent.run("and the year before?", session="s1")
    # no metric in the follow-up -> planner path; but entities must be inherited
    assert r["route"] != "refusal"


def test_step_cap_and_loop_guard(tmp_path, monkeypatch):
    step = {"tool": "get_financial_fact", "args": {"ticker": "NKE", "metric": "revenue", "fiscal_year": 2023}}
    agent = make_agent(by_role(router={"route": "narrative"}, planner={"steps": [step, step, step]},
                               answer={"answer": "Nike revenue $51,217 million [E1]", "citations": ["E1"]},
                               critic={"supported": True, "answers_question": True}), tmp_path)
    r = agent.run("Explain what drove Nike's revenue in 2023")
    facts = [p for p in r["plan"] if p["tool"] == "get_financial_fact"]
    assert len(facts) == 3 and "loop_detected" in r["failures"]
    assert r["plan"][0]["tool"] == "search_filings"  # led by a search with the question as asked


def test_tool_value_overrides_model_value(tmp_path):
    # live run: the model put its own 0.5489 in `value`; the tool had computed 0.5403
    agent = make_agent(by_role(answer={"answer": "General Mills retention ratio was 0.5489 [E1].", "value": 0.5489,
                                       "citations": ["E1"]}), tmp_path)
    r = agent.run("What is the FY2022 retention ratio (using total cash dividends paid and net income) for General Mills?")
    assert r["value"] == pytest.approx(0.5403, abs=1e-4)


def test_change_question_with_cashflow_spelling_uses_two_years(tmp_path):
    agent = make_agent(by_role(answer={"answer": "Adobe FCF conversion improved [E1][E2].", "citations": ["E1", "E2"]}),
                       tmp_path)  # no router/planner scripted: they must not be called
    r = agent.run("Does Adobe have an improving Free cashflow conversion as of FY2022?")
    assert [(p["args"]["ratio"], p["args"]["fiscal_year"]) for p in r["plan"]] == \
        [("fcf_conversion", 2022), ("fcf_conversion", 2021)]


@pytest.mark.parametrize("q", [
    "Which debt securities are registered to trade on a national securities exchange under American Express' name as of 2022?",
    "Are there any product categories / service categories that represent more than 20% of Boeing's revenue for FY2022?",
])
def test_categorical_questions_skip_lookup_fast_path(tmp_path, q):
    routed = []

    def router(user):
        routed.append(user)
        return {"route": "narrative"}

    agent = make_agent(by_role(router=router, planner={"steps": [{"tool": "search_filings", "args": {"query": "x"}}]},
                               answer={"answer": "See [E1].", "insufficient_evidence": True},
                               critic={"supported": True, "answers_question": True}), tmp_path)
    r = agent.run(q)
    assert routed and r["plan"][0]["purpose"] != "direct lookup"


def test_budget_overrun_returns_best_answer(tmp_path):
    agent = make_agent(by_role(answer={"answer": "Nike revenue was $55,000 million [E1].", "citations": ["E1"]}),
                       tmp_path)
    agent.llm.max_tokens_per_query = 100  # the fake client reports 120 tokens per call
    r = agent.run("What was Nike's revenue in FY2023?")
    assert "budget_exceeded" in r["failures"] and r["answer"].startswith("Nike revenue")
