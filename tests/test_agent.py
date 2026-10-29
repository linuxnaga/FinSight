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


def test_calculation_with_step_references(tmp_path):
    plan = {"steps": [
        {"tool": "get_financial_fact", "args": {"ticker": "NKE", "metric": "revenue", "fiscal_year": 2023}},
        {"tool": "get_financial_fact", "args": {"ticker": "NKE", "metric": "revenue", "fiscal_year": 2022}},
        {"tool": "calculate", "args": {"expression": "(a-b)/b*100", "variables": {"a": "$1", "b": "$2"}}},
    ]}

    def answer(user):
        assert "calculation" in user
        return {"answer": "Nike revenue grew 9.65% from FY2022 to FY2023 [E3].", "citations": ["E3"]}

    agent = make_agent(by_role(planner=plan, answer=answer), tmp_path)
    r = agent.run("What was Nike's revenue growth from FY2022 to FY2023?")
    assert r["route"] == "calculation"
    assert r["value"] == pytest.approx(9.65, abs=0.01)
    assert r["critiques"][0]["action"] == "accept"


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
    assert r["tool_calls"] == 1 and "loop_detected" in r["failures"]
