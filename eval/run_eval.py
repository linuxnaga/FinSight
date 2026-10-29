"""Run the agent on the gold set and write per-question results + a summary.

    python -m eval.run_eval                      # full 100
    python -m eval.run_eval --limit 30 --seed 1  # quota-friendly daily subset
    python -m eval.run_eval --no-judge           # skip LLM-as-judge (numeric + refusal only)

Outputs eval/results/<run>/results.jsonl and summary.md
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from finsight import llm as llm_mod
from finsight.agent.graph import Agent
from finsight.llm import LLM
from finsight.tracing import load_traces
from eval import metrics as M

HERE = Path(__file__).parent


def load_gold(limit: int | None = None, seed: int = 0, types: list[str] | None = None) -> list[dict]:
    items = [json.loads(l) for l in open(HERE / "gold.jsonl")]
    if types:
        items = [i for i in items if i["type"] in types]
    if limit:
        random.Random(seed).shuffle(items)
        items = items[:limit]
    return items


def retrieved_docs_from_trace(trace_id: str) -> list[str]:
    for t in load_traces(400):
        if t["trace_id"] == trace_id:
            ids = []
            for s in t["spans"]:
                if s["kind"] == "retrieval" and isinstance(s.get("output"), list):
                    ids.extend(cid.split("#")[0] for cid in s["output"])
            return ids
    return []


def score(item: dict, result: dict, judge_llm: LLM | None) -> dict:
    out: dict = {"refused": M.refused(result)}
    if item["should_refuse"]:
        out["correct"] = out["refused"]
        out["method"] = "refusal"
    elif item["type"] == "unanswerable":
        out["correct"] = M.said_not_found(result) and not out["refused"]
        out["method"] = "not_found"
    elif item["gold_value"] is not None:
        out["correct"] = M.numeric_match(result.get("value"), result.get("answer", ""), item["gold_value"],
                                         item.get("tolerance") or 0.01, M.gold_decimals(item["gold_answer"]))
        out["method"] = "numeric"
    elif judge_llm is not None:
        try:
            j = M.judge(judge_llm, item["question"], item["gold_answer"], result.get("answer", ""))
            out.update(correct=j.correct, judge_score=j.score, judge_reason=j.reason, method="judge")
        except Exception as e:
            out.update(correct=False, method="judge_error", judge_reason=str(e)[:200])
    else:
        out.update(correct=None, method="unscored")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--types", nargs="*")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"))
    a = ap.parse_args()

    items = load_gold(a.limit, a.seed, a.types)
    agent = Agent()
    judge_llm = None if a.no_judge else LLM()
    out_dir = HERE / "results" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_mod.STATS.update(calls=0, json_repairs=0, json_failures=0, mode_downgrades=0)

    rows = []
    with open(out_dir / "results.jsonl", "w") as f:
        for n, item in enumerate(items, 1):
            result = agent.run(item["question"])
            retrieved = retrieved_docs_from_trace(result["trace_id"])
            s = score(item, result, judge_llm)
            failure = None if s["correct"] in (True, None) else M.classify_failure(item, result, retrieved)
            rec = {"id": item["id"], "type": item["type"], "question": item["question"],
                   "gold": item["gold_answer"], "answer": result["answer"], "value": result.get("value"),
                   "route": result["route"], "tool_calls": result["tool_calls"], "latency_s": result["latency_s"],
                   "tokens": result["tokens"], "citation_hit": M.citation_hit(result, item["evidence_doc_ids"]),
                   "recall@5": M.recall_at_k(retrieved, item["evidence_doc_ids"], 5),
                   "failure": failure, "agent_failures": result["failures"], "trace_id": result["trace_id"], **s}
            rows.append(rec)
            f.write(json.dumps(rec) + "\n")
            print(f"[{n}/{len(items)}] {item['id']:24s} {'OK ' if s['correct'] else 'BAD' if s['correct'] is False else '-- '}"
                  f" {result['latency_s']:5.1f}s {result['tokens']:6d}tok  {failure or ''}")
    write_summary(rows, out_dir, dict(llm_mod.STATS))


def write_summary(rows: list[dict], out_dir: Path, stats: dict) -> None:
    scored = [r for r in rows if r["correct"] is not None]
    acc = sum(r["correct"] for r in scored) / max(1, len(scored))
    completed = [r for r in rows if not any(f in ("internal_error", "budget_exceeded") for f in r["agent_failures"])]
    lat = sorted(r["latency_s"] for r in rows)
    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        rs = [r for r in scored if r["type"] == t]
        if rs:
            by_type[t] = f"{sum(r['correct'] for r in rs)}/{len(rs)}"
    fails = Counter(r["failure"] for r in rows if r["failure"])
    rec = [r["recall@5"] for r in rows if r["recall@5"] is not None]
    cites = [r["citation_hit"] for r in rows if r["citation_hit"] is not None]
    lines = [
        f"# Eval run `{out_dir.name}`", "",
        "| metric | value |", "|---|---|",
        f"| end-to-end accuracy | {acc:.1%} ({sum(r['correct'] for r in scored)}/{len(scored)}) |",
        f"| task completion rate | {len(completed) / max(1, len(rows)):.1%} |",
        f"| avg tool calls / task | {statistics.mean(r['tool_calls'] for r in rows):.2f} |",
        f"| avg tokens / query | {statistics.mean(r['tokens'] for r in rows):.0f} |",
        f"| latency p50 / p95 (s) | {lat[len(lat) // 2]:.1f} / {lat[min(len(lat) - 1, int(len(lat) * .95))]:.1f} |",
        f"| retrieval recall@5 (agent searches) | {statistics.mean(rec):.2f} |" if rec else "| retrieval recall@5 | n/a |",
        f"| citation hit rate | {sum(cites) / len(cites):.1%} |" if cites else "| citation hit rate | n/a |",
        f"| LLM calls / JSON repairs / JSON failures | {stats['calls']} / {stats['json_repairs']} / {stats['json_failures']} |",
        "", "## Accuracy by question type", "", "| type | correct |", "|---|---|",
        *[f"| {t} | {v} |" for t, v in by_type.items()],
        "", "## Failure taxonomy", "", "| failure | count |", "|---|---|",
        *[f"| {k} | {fails.get(k, 0)} |" for k in M.FAILURE_TYPES],
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
