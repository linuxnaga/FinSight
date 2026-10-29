"""How far do the deterministic parts get with NO model at all?

Runs the agent with a stub LLM that fails every call. Whatever is still
answered correctly comes purely from rules, XBRL and the ratio tool. This is
the floor the 8B model builds on, and the number for the report's "what we did
to compensate for a small model" section (no endpoint or quota needed).

    python -m eval.deterministic_eval
"""
from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

from finsight.agent.graph import Agent
from finsight.agent.memory import SessionStore
from finsight.llm import StructuredOutputError
from eval import metrics as M
from eval.run_eval import load_gold

HERE = Path(__file__).parent


class NoLLM:
    """Every call fails, and is counted."""

    def __init__(self):
        self.calls = 0
        self.names: list[str] = []

    def structured(self, *a, name: str = "structured", **kw):
        self.calls += 1
        self.names.append(name)
        raise StructuredOutputError("offline: no model", "")

    def chat(self, *a, **kw):
        self.calls += 1
        raise RuntimeError("offline: no model")

    def text(self, *a, **kw):
        return self.chat()


def main() -> None:
    items = load_gold()
    llm = NoLLM()
    agent = Agent(llm=llm, store=SessionStore(Path(tempfile.mkdtemp()) / "s.sqlite"))
    rows, by_type = [], Counter()
    totals = Counter()
    for it in items:
        llm.names = []
        r = agent.run(it["question"])
        # did routing or the first plan need the model? (later calls are the answer and critic retries)
        first_answer = llm.names.index("answer") if "answer" in llm.names else len(llm.names)
        used_llm_before_answer = any(n in ("router", "planner") for n in llm.names[:first_answer])
        if it["should_refuse"]:
            ok = M.refused(r)
        elif it["gold_value"] is not None:
            ok = r.get("value") is not None and M.numeric_match(r["value"], "", it["gold_value"], it["tolerance"] or .01,
                                                                    M.gold_decimals(it["gold_answer"]))
        else:
            ok = None  # narrative: needs a model to write the answer
        rows.append({"id": it["id"], "type": it["type"], "correct": ok, "value": r.get("value"),
                     "gold": it["gold_value"], "route": r["route"], "plan": [p["tool"] for p in r["plan"]],
                     "llm_free_plan": not used_llm_before_answer, "refused": r.get("refused")})
        totals[it["type"]] += 1
        if ok:
            by_type[it["type"]] += 1
    scored = [r for r in rows if r["correct"] is not None]
    numeric = [r for r in scored if r["type"] not in ("refusal", "out_of_corpus", "injection")]
    safety = [r for r in scored if r["type"] in ("refusal", "out_of_corpus", "injection")]
    lines = [
        "# Deterministic floor (no LLM)", "",
        f"- numeric questions answered exactly: **{sum(r['correct'] for r in numeric)}/{len(numeric)}**",
        f"- refusal / out-of-corpus handled by rules: **{sum(r['correct'] for r in safety)}/{len(safety)}**",
        f"- questions routed and planned with zero LLM calls: **{sum(r['llm_free_plan'] for r in rows)}/{len(rows)}**",
        "", "| type | correct (of scorable) |", "|---|---|",
        *[f"| {t} | {by_type[t]}/{sum(1 for r in scored if r['type'] == t)} |"
          for t in sorted(totals) if any(r['type'] == t for r in scored)],
        "", "Misses on numeric questions:", "",
        *[f"- `{r['id']}` got {r['value']} vs gold {r['gold']} (plan: {', '.join(r['plan']) or 'none'})"
          for r in numeric if not r["correct"]],
    ]
    out = HERE / "results" / "deterministic_floor.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (HERE / "results" / "deterministic_floor.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
