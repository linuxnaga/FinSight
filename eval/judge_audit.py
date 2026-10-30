"""Measure how far the LLM judge can be trusted: a teammate labels a sample by
hand and we report agreement. The judge is the same 7B model being evaluated,
so the report should state its agreement with a human, not assume it.

    python -m eval.judge_audit export eval/results/baselines/fewshot_json__oracle.jsonl  # -> judge_audit.csv
    # open the CSV, fill the `human_correct` column with 1 or 0, save
    python -m eval.judge_audit score eval/results/judge_audit.csv
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from eval import metrics as M

HERE = Path(__file__).parent
FIELDS = ["id", "question", "gold_answer", "model_answer", "judge_correct", "figure_gate", "gated_correct",
          "human_correct"]


def export(paths: list[str], n: int = 30, seed: int = 0) -> Path:
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "gold.jsonl")}
    rows = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            g = gold[r["id"]]
            if g["gold_value"] is not None or g["type"] in ("unanswerable", "refusal", "out_of_corpus", "injection"):
                continue  # only judge-scored (narrative) items
            if r.get("correct") is None:
                continue
            judge = r.get("correct_judge_only", r["correct"])
            gate = M.figure_gate(g["gold_answer"], str(r.get("answer", "")))
            rows.append({"id": f"{Path(p).stem}:{r['id']}", "question": g["question"], "gold_answer": g["gold_answer"],
                         "model_answer": str(r.get("answer", ""))[:1500], "judge_correct": int(bool(judge)),
                         "figure_gate": "" if gate is None else int(gate),
                         "gated_correct": int(bool(judge) and gate is not False), "human_correct": ""})
    random.Random(seed).shuffle(rows)
    out = HERE / "results" / "judge_audit.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows[:n])
    print(f"{min(n, len(rows))} rows -> {out}\nFill `human_correct` with 1/0 (judge the answer against the gold, "
          f"ignoring the other columns), then run: python -m eval.judge_audit score {out}")
    return out


def score(path: str) -> None:
    rows = [r for r in csv.DictReader(open(path)) if r["human_correct"].strip() in ("0", "1")]
    if not rows:
        print("no labelled rows yet")
        return
    lines = [f"# Judge audit ({len(rows)} human-labelled answers)", "",
             "| scorer | agrees with human | false positives | false negatives |", "|---|---|---|---|"]
    for col, name in (("judge_correct", "LLM judge only"), ("gated_correct", "LLM judge + figure gate")):
        agree = sum(r[col] == r["human_correct"] for r in rows)
        fp = sum(r[col] == "1" and r["human_correct"] == "0" for r in rows)
        fn = sum(r[col] == "0" and r["human_correct"] == "1" for r in rows)
        lines.append(f"| {name} | {agree}/{len(rows)} ({agree / len(rows):.0%}) | {fp} | {fn} |")
    out = Path(path).with_suffix(".md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "export":
        export(sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        print(__doc__)
