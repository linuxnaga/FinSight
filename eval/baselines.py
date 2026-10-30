"""Week-1 baseline: single agent, single prompt, no tools.

Three prompting strategies x three context modes:

  strategies  fewshot_json   structured JSON output with 3 worked examples
              cot_sc         chain-of-thought, self-consistency (n samples, majority vote on the number)
              decompose      first list sub-questions, then answer them, then combine
  context     closed         no documents (what does the model "know"?)
              oracle         the gold evidence section(s), truncated to fit
              naive_rag      top-5 BM25 chunks with company/year filters

The gap closed -> oracle shows how much a weak generator gains from perfect
retrieval; oracle -> naive_rag shows what retrieval loses. Document HOW the
model breaks in eval/results/baselines/breakage_log.md as you read outputs.

    python -m eval.baselines --strategies fewshot_json cot_sc decompose --modes closed oracle --limit 40
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from finsight.config import SECTIONS_DIR
from finsight.llm import LLM, StructuredOutputError
from eval import metrics as M
from eval.run_eval import load_gold

HERE = Path(__file__).parent


class BaseAnswer(BaseModel):
    answer: str
    value: float | None = None


FEWSHOT = """Answer questions about company 10-K filings.
Return JSON {"answer": "...", "value": <headline number or null>}.

Example 1: Q: What was 3M's FY2022 capital expenditure in USD millions?
{"answer": "3M's FY2022 capex (purchases of PP&E) was $1,749 million.", "value": 1749}
Example 2: Q: What was Nike's FY2023 gross margin?
{"answer": "Gross margin was 43.5% (gross profit 22,292 / revenue 51,217).", "value": 43.5}
Example 3: Q: What does PepsiCo say about supply chain risk in FY2022?
{"answer": "PepsiCo cites disruption of raw materials, packaging and transportation, inflation in input costs ...", "value": null}"""

COT = """Answer questions about company 10-K filings. Think step by step: identify the figures needed,
write them down with their source, do the arithmetic carefully, then give the final answer.
End with a line exactly: FINAL: <answer> | VALUE: <number or none>"""

DECOMPOSE_1 = "Break the question into 1-4 short sub-questions needed to answer it. Return JSON {\"subs\": [..]}."
DECOMPOSE_2 = """Answer each sub-question, then combine them into a final answer.
End with a line exactly: FINAL: <answer> | VALUE: <number or none>"""


class Subs(BaseModel):
    subs: list[str]


def _context(item: dict, mode: str, sections: dict[str, str], limit: int = 9000) -> str:
    if mode == "closed":
        return ""
    if mode == "oracle":
        # the exact evidence (FinanceBench pages / reported XBRL figures). Gold sections are ~160k
        # characters, so a section prefix almost never contains the answer.
        text = "\n\n".join(item.get("evidence_text") or [])[:limit]
        if not text:
            text = "\n\n".join(sections.get(d, "")[: limit // max(1, len(item["evidence_doc_ids"]))]
                               for d in item["evidence_doc_ids"])
    else:
        from finsight.retrieval.index import default_index

        t = item["ticker"] if isinstance(item["ticker"], str) else None
        hits = default_index().search(item["question"], k=5, ticker=item["ticker"], fiscal_year=item["fiscal_year"],
                                      use_dense=False, use_rerank=False)
        text = "\n\n".join(h.chunk.text for h in hits)[:limit]
    return f"\n\nFiling excerpts:\n{text}" if text else ""


def _parse_final(text: str) -> tuple[str, float | None]:
    m = re.search(r"FINAL:\s*(.*?)\s*\|\s*VALUE:\s*([^\n]*)", text, re.S)
    if not m:
        return text.strip()[-600:], None
    ans, val = m.group(1).strip(), m.group(2).strip()
    try:
        return ans, float(val.replace(",", "").replace("$", "").replace("%", ""))
    except ValueError:
        return ans, None


def run_one(llm: LLM, strategy: str, item: dict, ctx: str, n_sc: int) -> dict:
    q = item["question"] + ctx
    if strategy == "fewshot_json":
        try:
            a = llm.structured(FEWSHOT, q, BaseAnswer, name="base_fewshot", retries=1)
            return {"answer": a.answer, "value": a.value, "json_ok": True}
        except StructuredOutputError as e:
            return {"answer": e.raw, "value": None, "json_ok": False}
    if strategy == "cot_sc":
        outs = llm.chat([{"role": "system", "content": COT}, {"role": "user", "content": q}],
                        name="base_cot", temperature=0.7, max_tokens=700, n=n_sc)
        if len(outs) < n_sc:  # endpoint ignored n -> sample sequentially
            outs += [llm.chat([{"role": "system", "content": COT}, {"role": "user", "content": q}],
                              name="base_cot", temperature=0.7, max_tokens=700)[0] for _ in range(n_sc - len(outs))]
        parsed = [_parse_final(o) for o in outs]
        vals = [round(v, 1) for _, v in parsed if v is not None]
        if vals:
            top = Counter(vals).most_common(1)[0][0]
            ans = next(a for a, v in parsed if v is not None and round(v, 1) == top)
            return {"answer": ans, "value": top, "votes": dict(Counter(vals))}
        return {"answer": parsed[0][0], "value": None}
    if strategy == "decompose":
        try:
            subs = llm.structured(DECOMPOSE_1, item["question"], Subs, name="base_decompose", retries=1).subs
        except StructuredOutputError:
            subs = [item["question"]]
        user = "Question: " + item["question"] + "\nSub-questions:\n- " + "\n- ".join(subs) + ctx
        ans, val = _parse_final(llm.text(DECOMPOSE_2, user, name="base_decompose_2", max_tokens=700))
        return {"answer": ans, "value": val, "subs": subs}
    raise ValueError(strategy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=["fewshot_json", "cot_sc", "decompose"])
    ap.add_argument("--modes", nargs="*", default=["closed", "oracle"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--n-sc", type=int, default=5)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()

    items = [i for i in load_gold(a.limit, 0) if not i["should_refuse"]]
    sections = {json.loads(l)["doc_id"]: json.loads(l)["text"] for l in open(SECTIONS_DIR / "sections.jsonl")}
    llm, judge_llm = LLM(), (None if a.no_judge else LLM())
    out_dir = HERE / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for strategy in a.strategies:
        for mode in a.modes:
            path = out_dir / f"{strategy}__{mode}.jsonl"
            correct = scored = 0
            t0 = time.time()
            with open(path, "w") as f:
                for item in items:
                    try:
                        r = run_one(llm, strategy, item, _context(item, mode, sections), a.n_sc)
                    except Exception as e:
                        r = {"answer": f"ERROR {e}", "value": None}
                    if item["gold_value"] is not None:
                        ok = M.numeric_match(r.get("value"), r["answer"], item["gold_value"], item["tolerance"] or .01,
                                             M.gold_decimals(item["gold_answer"]))
                    elif item["type"] == "unanswerable":
                        ok = M.said_not_found(r)
                    elif judge_llm:
                        try:
                            ok = M.judge(judge_llm, item["question"], item["gold_answer"], r["answer"]).correct
                        except Exception:
                            ok = False
                    else:
                        ok = None
                    if ok is not None:
                        scored += 1
                        correct += ok
                    f.write(json.dumps({"id": item["id"], "type": item["type"], "correct": ok, **r}) + "\n")
            row = (strategy, mode, correct, scored, time.time() - t0)
            table.append(row)
            print(f"{strategy:13s} {mode:9s} {correct}/{scored} = {correct / max(1, scored):.1%}")
    md = ["| strategy | context | accuracy |", "|---|---|---|"] + \
         [f"| {s} | {m} | {c}/{n} ({c / max(1, n):.0%}) |" for s, m, c, n, _ in table]
    (out_dir / "baseline_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
