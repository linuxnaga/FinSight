"""The agent: guard -> route -> plan -> execute -> answer -> critic (-> revise).

    guard     rules: refusal policy, PII masking, entities, follow-up resolution
    route     LLM classifier (enum output)
    plan      deterministic fast path for single-metric lookups, else LLM planner
    execute   validated tool calls, "$N" references, fallbacks, loop detection
    answer    LLM writes a cited answer from labelled evidence only
    critic    deterministic number/citation checks + LLM groundedness check
              for narrative answers; up to MAX_CRITIC_REVISIONS rewrite/replan

Hard limits: MAX_STEPS tool calls and MAX_TOKENS_PER_QUERY per question.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from finsight import tracing
from finsight.agent import guard, prompts
from finsight.agent.memory import SessionStore
from finsight.agent.schemas import Answer, Critique, Plan, PlanStep, Route
from finsight.config import COMPANIES, FISCAL_YEARS, settings
from finsight.llm import LLM, BudgetExceeded, StructuredOutputError
from finsight.tools import ratios, xbrl
from finsight.tools.registry import run_tool, tool_catalog, validate_args

DISCLAIMER = "Draft generated from SEC filings by an AI system - verify figures against the source before use."


@dataclass
class Evidence:
    eid: str
    kind: str  # "fact" | "passage" | "calc"
    text: str
    source: str
    url: str | None = None
    numbers: list[float] = field(default_factory=list)


@dataclass
class State:
    question: str
    session: str
    entities: guard.Entities
    route: str = ""
    plan: list[PlanStep] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)  # per executed step
    evidence: list[Evidence] = field(default_factory=list)
    answer: Answer | None = None
    critiques: list[dict] = field(default_factory=list)
    steps_used: int = 0
    failures: list[str] = field(default_factory=list)
    injection_flags: int = 0
    ratio_fast_path: bool = False


# ------------------------------------------------------------------ helpers
_NUM = re.compile(r"(?<![\w.])-?\$?\(?\d[\d,]*\.?\d*\)?%?")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        s = m.replace("$", "").replace(",", "").replace("%", "").replace("(", "").replace(")", "")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def _is_supported(x: float, pool: list[float]) -> bool:
    if abs(x) < 32 and float(x).is_integer():  # small ordinals, day numbers, "3 segments"
        return True
    if 1990 <= x <= 2035 and float(x).is_integer():  # years
        return True
    for p in pool:
        for scale in (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
            v = p / scale
            if v and abs(x - v) / abs(v) < 0.006:  # tolerate rounding to 2-3 significant digits
                return True
    return False


_BOILERPLATE = re.compile(r"(statements? of cash flows?|cash flows? statements?|income statements?|balance sheets?|"
                         r"statements? of (financial position|income|operations|earnings)|p&l statements?)", re.I)
# "free cashflow", "cash-flow" -> "cash flow" before metric/ratio detection
def normalize_terms(text: str) -> str:
    return re.sub(r"\bcash[\s-]?flows?\b", "cash flow", text, flags=re.I)


# yes/no, "which"/"who" and listing questions want facts + text, not a single looked-up number
CATEGORICAL_RE = re.compile(r"^\s*(are|is|was|were|does|do|did|has|have|had|can|which|who|what (are|were) the)\b|"
                            r"\b(are there|is there|which|list|name the|any (product|service|segment))\b", re.I)
# a ratio asked about over time ("improving", "declined") needs the prior year too
CHANGE_RE = re.compile(r"\b(improv\w*|declin\w*|deteriorat\w*|chang\w*|increas\w*|decreas\w*|"
                       r"grew|grow\w*|trend\w*|vs\.? (the )?prior)\b", re.I)

LEAD_PURPOSE = "search with the question as asked"

# questions asking for an explanation need text, not just a number
WHY_RE = re.compile(r"\b(why|explain\w*|reasons?|drivers?|drove|driven|how did|describe|tell me about|cause[sd]?|"
                    r"primarily due)\b", re.I)
# concept names too generic to match on their own ("cash" appears in "cash dividends")
_NEEDS_ALIAS = {"cash"}
_CASH_RE = re.compile(r"\b(cash and (cash )?equivalents|cash balance|cash on hand|cash position)\b", re.I)


def detect_metrics(question: str) -> list[str]:
    """Metrics mentioned in the question, in order of appearance, longest
    phrase wins (so "cash dividends" is dividends, not cash)."""
    q = _BOILERPLATE.sub(" ", normalize_terms(question).lower())
    names = {**{k.replace("_", " "): k for k in xbrl.CONCEPTS if k not in _NEEDS_ALIAS}, **xbrl.ALIASES}
    hits: list[tuple[int, int, str]] = []
    for alias in sorted(names, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w]){re.escape(alias)}(?![\w])", q):
            if not any(a <= m.start() < b or a < m.end() <= b for a, b, _ in hits):
                hits.append((m.start(), m.end(), names[alias]))
    if _CASH_RE.search(q):
        hits.append((_CASH_RE.search(q).start(), 0, "cash"))
    found: list[str] = []
    for _, _, key in sorted(hits):
        if key not in found:
            found.append(key)
    return found


class Agent:
    def __init__(self, llm: LLM | None = None, store: SessionStore | None = None):
        self._llm = llm
        self._state: State | None = None
        self.store = store or SessionStore()
        self.s = settings()

    @property
    def llm(self) -> LLM:
        # created on first use, so rule-based refusals work even before the endpoint is configured
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    # ============================================================== public
    def run(self, question: str, session: str | None = None) -> dict:
        session = session or uuid.uuid4().hex[:8]
        t0 = time.time()
        with tracing.trace(question) as tr:
            try:
                result = self._run(question, session)
            except BudgetExceeded as e:
                st = self._state
                if st is not None and st.answer is not None and st.answer.answer:
                    # out of budget mid-revision: return the best answer so far, flagged
                    st.failures.append("budget_exceeded")
                    result = self._finish(st)
                else:
                    result = self._final_error(f"Stopped: {e}", "budget_exceeded")
            except Exception as e:  # last-resort guard: the UI always gets an answer object
                result = self._final_error(f"Internal error: {type(e).__name__}: {e}", "internal_error")
            result.update(trace_id=tr.trace_id, session=session, latency_s=round(time.time() - t0, 2),
                          tokens=tr.tokens)
            tr.answer = {k: result.get(k) for k in ("answer", "value", "unit", "route", "refused", "failures")}
        return result

    # ============================================================ pipeline
    def _run(self, question: str, session: str) -> dict:
        self._state = None
        history = self.store.history(session)
        prev = guard.Entities(**history[-1]["entities"]) if history else None

        # ---- guard (deterministic)
        with tracing.span("guard", "node", input=question) as sp:
            masked, pii = guard.mask_pii(question)
            reason = guard.refusal_reason(masked)
            ents = guard.extract_entities(masked, prev)
            sp.output = {"pii": pii, "refusal": reason, "tickers": ents.tickers, "years": ents.years,
                         "outside": ents.outside, "inherited": ents.inherited}
        st = State(question=masked, session=session, entities=ents)
        self._state = st
        if reason:
            return self._finish(st, guard.REFUSAL_TEXT[reason], refused=reason)
        if ents.outside and not ents.tickers:
            return self._finish(st, f"{', '.join(n.title() for n in ents.outside)} is not in my corpus. I cover: "
                                    f"{', '.join(v[1] for v in COMPANIES.values())} (FY{FISCAL_YEARS[0]}-"
                                    f"FY{FISCAL_YEARS[-1]} 10-Ks).", refused="out_of_corpus")
        if ents.tickers and not ents.years:
            # no year given and nothing to inherit: use the latest filing, and say so in the answer
            ents.years, ents.assumed_year = [FISCAL_YEARS[-1]], True
        bad_years = guard.out_of_range_years(ents)
        if bad_years and len(bad_years) == len(ents.years):
            return self._finish(st, f"FY{bad_years[0]} is outside the corpus, which covers FY{FISCAL_YEARS[0]}-"
                                    f"FY{FISCAL_YEARS[-1]} 10-K filings.", refused="out_of_corpus")

        # ---- route
        st.route = self._route(st)
        if st.route == "out_of_scope" and not ents.tickers:
            return self._finish(st, "That question isn't about the 10-K filings I cover. Ask me about the reported "
                                    "financials, risks or business of the covered companies.", refused="out_of_scope")

        # ---- plan / execute / answer / critic loop
        st.plan = self._plan(st, history)
        self._execute(st)
        st.answer = self._answer(st)
        for revision in range(self.s.max_critic_revisions):
            crit = self._critic(st)
            st.critiques.append(crit.model_dump())
            if crit.action == "accept":
                break
            if crit.action == "replan" and st.steps_used < self.s.max_steps:
                st.plan = self._plan(st, history, feedback=crit.issues)
                self._execute(st)
            st.answer = self._answer(st, feedback=crit.issues)
        return self._finish(st)

    # --------------------------------------------------------------- route
    def _route(self, st: State) -> str:
        with tracing.span("route", "node", input=st.question) as sp:
            metrics = detect_metrics(st.question)
            n_t, n_y = len(st.entities.tickers), len(st.entities.years)
            # Cheap deterministic routing for the unambiguous cases.
            if self._ratio_plan(st):
                route = "comparison" if n_t > 1 else "calculation"
            elif n_t > 1 or (n_y > 1 and metrics and re.search(r"\b(grow|growth|change|increase|decrease|from)\b", st.question.lower())):
                route = "comparison" if n_t > 1 else "calculation"
            elif metrics and re.search(r"\b(margin|ratio|growth|grew|change|per cent|percent|%|turnover|"
                                       r"return on|yoy|difference)\b", st.question.lower()):
                route = "calculation"
            elif len(metrics) == 1 and n_t == 1 and n_y == 1 and not WHY_RE.search(st.question) and \
                    not CATEGORICAL_RE.search(st.question) and not re.search(
                    r"\b(per|ratio|divided|calculate|compute|times|average|versus|vs|compared)\b", st.question.lower()):
                route = "lookup"
            else:
                try:
                    r = self.llm.structured(prompts.ROUTER, st.question, Route, name="router", max_tokens=80)
                    route = r.route
                except StructuredOutputError:
                    route = "narrative"
                    st.failures.append("tool_call_error:router_json")
                if route == "out_of_scope" and st.entities.tickers:
                    route = "narrative"
                if route == "narrative" and metrics and n_t == 1 and n_y == 1 and len(metrics) == 1 and \
                        not WHY_RE.search(st.question) and not CATEGORICAL_RE.search(st.question):
                    route = "lookup"
            sp.output = {"route": route, "metrics": metrics}
        return route

    # ---------------------------------------------------------------- plan
    def _plan(self, st: State, history: list[dict], feedback: list[str] | None = None) -> list[PlanStep]:
        ents = st.entities
        metrics = detect_metrics(st.question)
        ratio_steps = None if feedback else self._ratio_plan(st)
        if ratio_steps:
            st.ratio_fast_path = True
            with tracing.span("plan", "node", input="ratio fast-path") as sp:
                sp.output = [p.model_dump() for p in ratio_steps]
            return ratio_steps
        # Fast path: single-metric lookup needs no LLM planning at all.
        if not feedback and st.route == "lookup" and len(ents.tickers) == 1 and len(ents.years) == 1 \
                and len(metrics) == 1:
            plan = [PlanStep(tool="get_financial_fact",
                             args={"ticker": ents.tickers[0], "metric": metrics[0], "fiscal_year": ents.years[0]},
                             purpose="direct lookup")]
            with tracing.span("plan", "node", input="fast-path") as sp:
                sp.output = [p.model_dump() for p in plan]
            return plan

        ctx = [f"Question: {st.question}",
               f"Companies detected: {ents.tickers or 'none'}; fiscal years detected: {ents.years or 'none'}"
               + (" (inherited from the previous turn)" if ents.inherited else "")]
        if history:
            ctx.append("Previous turn: " + history[-1]["question"])
        if feedback:
            ctx.append("Your previous plan was insufficient. Problems: " + "; ".join(feedback[:4]))
            ctx.append("Evidence already gathered: " + "; ".join(e.text[:120] for e in st.evidence[:6]))
        system = prompts.PLANNER.format(catalog=tool_catalog())
        user = "\n".join(ctx)
        with tracing.span("plan", "node", input=user) as sp:
            plan: list[PlanStep] = []
            for attempt in range(2):
                try:
                    p = self.llm.structured(system, user, Plan, name="planner", max_tokens=500)
                except StructuredOutputError:
                    st.failures.append("tool_call_error:planner_json")
                    break
                errors = []
                for i, step in enumerate(p.steps, 1):
                    try:
                        args = dict(step.args)
                        if step.tool == "calculate":  # "$N" refs are resolved at execution time
                            args["variables"] = {k: 0 for k in args.get("variables", {})}
                        validate_args(step.tool, args)
                    except (ValueError, ValidationError) as e:
                        errors.append(f"step {i} ({step.tool}): {str(e).splitlines()[0]} {str(e)[-200:]}")
                if not errors:
                    plan = p.steps
                    break
                st.failures.append("tool_call_error:bad_args")
                user = user + "\n\nYour plan had invalid tool arguments:\n" + "\n".join(errors) + "\nFix them."
            if not plan:
                plan = self._fallback_plan(st, metrics)
            plan = self._with_question_search(st, plan) if not feedback else plan
            sp.output = [p.model_dump() for p in plan]
        return plan

    def _with_question_search(self, st: State, plan: list[PlanStep]) -> list[PlanStep]:
        """Lead with a search on the user's own words. The planner's paraphrases ("acquisition major
        deal") retrieved the gold section far less often than the question itself (recall@5 0.92
        offline vs 0.12 in the agent), so its passages come first in the evidence."""
        ents = st.entities
        if not ents.tickers:
            return plan
        years = ents.years[-2:] if len(ents.tickers) == 1 else ents.years[-1:]
        lead = [PlanStep(tool="search_filings", args={"query": st.question, "ticker": t, "fiscal_year": y, "k": 5},
                         purpose=LEAD_PURPOSE)
                for t in ents.tickers[:2] for y in (years or [None])][:2]
        return (lead + plan)[: self.s.max_steps]

    def _ratio_plan(self, st: State) -> list[PlanStep] | None:
        """Deterministic plan for standard ratios and single-metric YoY growth.
        Returns None when the question doesn't fit (the LLM planner takes over)."""
        ents, q = st.entities, st.question.lower()
        if not ents.tickers or not ents.years:
            return None
        if re.search(r"\b(cagr|over the (last|past)|trend|\d[- ]year|two[- ]year|three[- ]year)\b", q) or \
                max(ents.years) - min(ents.years) > 1:
            return None  # multi-year aggregates -> LLM planner ("average X" inside a ratio definition is fine)
        found = ratios.detect_ratios(st.question)
        steps: list[PlanStep] = []
        years = list(ents.years)
        if len(years) == 1 and CHANGE_RE.search(q) and years[0] - 1 >= FISCAL_YEARS[0] - 2:
            years = [years[0] - 1, years[0]]  # "improving / declined as of FY2022" compares with FY2021
        if found:
            for t in ents.tickers[:3]:
                for r in found[:2]:
                    for y in sorted(years[-2:], reverse=True):  # asked-about year first
                        steps.append(PlanStep(tool="compute_ratio", args={"ticker": t, "ratio": r, "fiscal_year": y},
                                              purpose=f"{r} {t} FY{y}"))
        else:
            metrics = detect_metrics(st.question)
            years = ents.years
            consecutive = len(years) == 1 or (len(years) == 2 and years[1] - years[0] == 1)
            if len(metrics) == 1 and ratios.GROWTH_RE.search(q) and consecutive:
                for t in ents.tickers[:3]:
                    steps.append(PlanStep(tool="compute_ratio", args={"ticker": t, "ratio": "growth",
                                          "metric": metrics[0], "fiscal_year": years[-1]},
                                          purpose=f"{metrics[0]} growth {t} FY{years[-1]}"))
            if not steps and len(metrics) == 1 and len(ents.tickers) > 1:
                steps.append(PlanStep(tool="compare_peers", args={"tickers": ents.tickers[:6], "metric": metrics[0],
                                      "fiscal_years": ents.years[-4:]}, purpose=f"{metrics[0]} across companies"))
        if not steps:
            return None
        if WHY_RE.search(q):
            steps.append(PlanStep(tool="search_filings", args={
                "query": st.question, "ticker": ents.tickers[0], "fiscal_year": ents.years[-1], "item": "7", "k": 4},
                purpose="management's explanation"))
        return steps[:6]

    def _fallback_plan(self, st: State, metrics: list[str]) -> list[PlanStep]:
        """When the planner can't produce a valid plan, search with whatever we know."""
        ents = st.entities
        return [PlanStep(tool="search_filings", args={
            "query": st.question, "ticker": ents.tickers[0] if len(ents.tickers) == 1 else None,
            "fiscal_year": ents.years[0] if len(ents.years) == 1 else None, "k": 6}, purpose="fallback search")]

    # ------------------------------------------------------------- execute
    def _execute(self, st: State) -> None:
        seen = {json.dumps([o["tool"], o["args"]], sort_keys=True, default=str) for o in st.outputs}
        base = len(st.outputs)
        for i, step in enumerate(st.plan, 1):
            if st.steps_used >= self.s.max_steps:
                st.failures.append("step_cap_reached")
                break
            args = self._resolve_refs(step, st, base)
            key = json.dumps([step.tool, args], sort_keys=True, default=str)
            if key in seen:  # identical call already made -> loop guard
                st.failures.append("loop_detected")
                st.outputs.append({"tool": step.tool, "args": args, "out": {"skipped": "duplicate call"},
                                   "lead": step.purpose == LEAD_PURPOSE})
                continue
            seen.add(key)
            st.steps_used += 1
            out = run_tool(step.tool, args)
            if "error" in out and step.tool == "get_financial_fact" and st.steps_used < self.s.max_steps:
                st.failures.append("tool_call_error:fact_missing")
                st.steps_used += 1
                fb_args = {"query": f"{args.get('metric', '').replace('_', ' ')} consolidated statement",
                           "ticker": args.get("ticker"), "fiscal_year": args.get("fiscal_year"), "item": "8", "k": 4}
                out = {**out, "fallback": run_tool("search_filings", fb_args)}
            st.outputs.append({"tool": step.tool, "args": args, "out": out, "lead": step.purpose == LEAD_PURPOSE})
            self._collect_evidence(st, step.tool, out)

    def _resolve_refs(self, step: PlanStep, st: State, base: int) -> dict:
        args = json.loads(json.dumps(step.args, default=str))
        if step.tool != "calculate":
            return args
        resolved = {}
        for name, v in args.get("variables", {}).items():
            if isinstance(v, str) and re.fullmatch(r"\$\d+", v.strip()):
                n = int(v.strip()[1:])
                # "$N" is the planner's own step N: the lead question searches don't count
                own = [o for o in st.outputs[base:] if not o.get("lead")]
                ref = own[n - 1]["out"] if 0 < n <= len(own) else {}
                if "value_millions" in ref and ref["value_millions"] is not None:
                    resolved[name] = ref["value_millions"]
                elif "value" in ref:
                    resolved[name] = ref["value"]
                else:
                    resolved[name] = v  # leave unresolved -> tool returns a validation error
            else:
                resolved[name] = v
        args["variables"] = resolved
        return args

    def _collect_evidence(self, st: State, tool: str, out: dict) -> None:
        def add(kind, text, source, url=None, nums=None):
            eid = f"E{len(st.evidence) + 1}"
            st.evidence.append(Evidence(eid, kind, text, source, url, nums if nums is not None else numbers_in(text)))

        if "error" in out and "fallback" not in out:
            return
        if tool == "get_financial_fact" and "value" in out:
            v = out["value_millions"] if out.get("value_millions") is not None else out["value"]
            unit = "USD millions" if out.get("value_millions") is not None else out["unit"]
            add("fact", f"{out['company']} FY{out['fiscal_year']} {out['metric'].replace('_', ' ')} = {v:,} {unit} "
                        f"(period {out.get('period_start') or ''}..{out['period_end']})",
                out["source"], None, [float(v), float(out["value"])])
        elif tool == "compare_peers":
            for r in out.get("rows", []):
                v = r["value"] / 1e6 if r["unit"] == "USD" else r["value"]
                u = "USD millions" if r["unit"] == "USD" else r["unit"]
                add("fact", f"{COMPANIES[r['ticker']][1]} FY{r['fiscal_year']} {out['metric'].replace('_', ' ')} "
                            f"= {v:,.2f} {u}", r["source"], None, [v, float(r["value"])])
        elif tool == "compute_ratio" and "value" in out:
            parts = "; ".join(f"{k} = {v['value']:,.2f} {v['unit']}" for k, v in out["inputs"].items())
            label = out["ratio"].replace("_", " ") if out["ratio"] != "growth" else \
                f"{(out.get('metric') or '').replace('_', ' ')} YoY growth"
            add("calc", f"{out['company']} FY{out['fiscal_year']} {label} = {out['value']:,.2f} {out['unit']} "
                        f"(definition: {out['definition']}; inputs: {parts})",
                "; ".join(sorted({v["source"] for v in out["inputs"].values()})), None,
                [float(out["value"])] + [float(v["value"]) for v in out["inputs"].values()])
        elif tool == "calculate" and "value" in out:
            add("calc", f"calculation {out['expression']} with {out['variables']} = {out['value']:,.4f}",
                "calculator", None, [float(out["value"])])
        passages = out.get("passages") or out.get("fallback", {}).get("passages") or []
        for p in passages:
            text, flagged = guard.screen_passage(p["text"])
            if flagged:
                st.injection_flags += 1
                tracing.current() and tracing.current().spans and setattr(
                    tracing.current().spans[-1], "failure", "prompt_injection_blocked")
            add("passage", text, p["doc_id"], p.get("url"))

    # -------------------------------------------------------------- answer
    def _evidence_block(self, st: State, limit_chars: int = 9000) -> str:
        parts, used = [], 0
        # facts/calcs first: they're short and authoritative
        for e in sorted(st.evidence, key=lambda e: {"calc": 0, "fact": 1, "passage": 2}[e.kind]):
            block = f"[{e.eid}] ({e.source}) {e.text}"
            if used + len(block) > limit_chars:
                block = block[: max(0, limit_chars - used)]
            if not block:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts) or "(no evidence found)"

    def _answer(self, st: State, feedback: list[str] | None = None) -> Answer:
        user = f"Question: {st.question}\n\nEvidence:\n{self._evidence_block(st)}"
        if feedback:
            user += "\n\nA reviewer found problems in your previous draft. Fix them:\n- " + "\n- ".join(feedback[:5])
        with tracing.span("answer", "node") as sp:
            try:
                a = self.llm.structured(prompts.ANSWER, user, Answer, name="answer", max_tokens=600)
            except StructuredOutputError as e:
                st.failures.append("tool_call_error:answer_json")
                a = Answer(answer=e.raw[:1200] or "I could not produce a reliable answer.",
                           insufficient_evidence=not e.raw)
            # deterministic headline: when a tool computed the number, it wins over the model's own
            # (a model-filled value of 0.5489 once replaced the tool's correct 0.5403)
            calcs = [e for e in st.evidence if e.kind == "calc"]
            facts = [e for e in st.evidence if e.kind == "fact"]
            if st.route in ("lookup", "calculation"):
                if calcs:
                    # ratio fast path: first step is the ratio asked for; planner chains end in the result
                    primary = calcs[0] if st.ratio_fast_path else calcs[-1]
                    a.value = primary.numbers[0]
                elif len(facts) == 1:
                    a.value = facts[0].numbers[0]
            sp.output = a.model_dump()
        return a

    # -------------------------------------------------------------- critic
    def _critic(self, st: State) -> Critique:
        a = st.answer
        assert a is not None
        with tracing.span("critic", "node") as sp:
            issues: list[str] = []
            ids = {e.eid for e in st.evidence}
            pool = [n for e in st.evidence for n in e.numbers]
            if not st.evidence:
                issues.append("no evidence was retrieved")
            bad_cites = [c for c in a.citations if c.strip("[]") not in ids]
            if bad_cites:
                issues.append(f"citations {bad_cites} do not exist")
            if st.evidence and not a.insufficient_evidence and not re.search(r"\[E\d+\]", a.answer) and not a.citations:
                issues.append("answer has no citations")
            unsupported = [x for x in numbers_in(a.answer) if not _is_supported(x, pool)]
            if a.value is not None and not _is_supported(a.value, pool):
                unsupported.append(a.value)
            if unsupported:
                issues.append(f"numbers not found in evidence: {unsupported[:5]}")
            for t in st.entities.tickers:
                if COMPANIES[t][1].lower().split()[0] not in a.answer.lower() and t.lower() not in a.answer.lower():
                    issues.append(f"answer does not mention {COMPANIES[t][1]}")
            crit: Critique
            if issues:
                action = "replan" if (not st.evidence or a.insufficient_evidence) else "rewrite"
                crit = Critique(supported=False, answers_question=True, issues=issues, action=action)
                if unsupported:
                    sp.failure = "hallucination"
            elif st.route == "narrative":
                # narrative claims can't be checked with regexes -> LLM groundedness check
                user = (f"Question: {st.question}\n\nEvidence:\n{self._evidence_block(st, 6000)}\n\n"
                        f"Draft answer:\n{a.answer}")
                try:
                    crit = self.llm.structured(prompts.CRITIC, user, Critique, name="critic_llm", max_tokens=250)
                except StructuredOutputError:
                    crit = Critique(supported=True, answers_question=True, issues=["critic unavailable"])
                if not crit.supported:
                    sp.failure = "hallucination"
            else:
                crit = Critique(supported=True, answers_question=True)
            if a.insufficient_evidence and crit.action == "accept" and st.steps_used < self.s.max_steps \
                    and len(st.critiques) == 0:
                crit.action, crit.issues = "replan", crit.issues + ["answer says evidence is insufficient"]
            sp.output = crit.model_dump()
        return crit

    # -------------------------------------------------------------- finish
    def _finish(self, st: State, text: str | None = None, refused: str | None = None) -> dict:
        a = st.answer
        used = set(re.findall(r"E\d+", (a.answer if a else "") + " ".join(a.citations if a else [])))
        citations = [{"id": e.eid, "source": e.source, "url": e.url, "snippet": e.text[:400]}
                     for e in st.evidence if e.eid in used]
        answer_text = text if text is not None else (a.answer if a else "")
        if st.entities.assumed_year and not refused:
            answer_text = f"(No fiscal year given; using FY{st.entities.years[0]}, the latest in the corpus.) " + answer_text
        result = {
            "question": st.question,
            "answer": answer_text,
            "value": None if text is not None or a is None else a.value,
            "unit": None if text is not None or a is None else a.unit,
            "insufficient_evidence": bool(a and a.insufficient_evidence),
            "citations": citations,
            "route": st.route or ("refusal" if refused else ""),
            "refused": refused,
            "plan": [p.model_dump() for p in st.plan],
            "tool_calls": st.steps_used,
            "critiques": st.critiques,
            "failures": st.failures,
            "injection_flags": st.injection_flags,
            "disclaimer": DISCLAIMER,
        }
        self.store.append(st.session, {"question": st.question, "answer": result["answer"][:500],
                                       "entities": st.entities.__dict__})
        return result

    @staticmethod
    def _final_error(msg: str, tag: str) -> dict:
        return {"answer": msg, "value": None, "unit": None, "citations": [], "route": "", "refused": None,
                "plan": [], "tool_calls": 0, "critiques": [], "failures": [tag], "injection_flags": 0,
                "insufficient_evidence": True, "disclaimer": DISCLAIMER}
