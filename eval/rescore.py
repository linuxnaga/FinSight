"""Re-apply the deterministic figure gate to saved results, without calling
the model. Use it on runs made before the gate existed, or to show the
judge-only vs gated numbers side by side in the report.

    python -m eval.rescore                          # eval/results/baselines/*.jsonl
    python -m eval.rescore eval/results/<run>/results.jsonl
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

from eval import metrics as M

HERE = Path(__file__).parent


def rescore_file(path: str, gold: dict[str, dict]) -> tuple[int, int, int, int]:
    """Returns (judge_correct, gated_correct, scored, flipped) and rewrites the
    file with `correct_judge_only` kept alongside the gated `correct`."""
    rows = [json.loads(l) for l in open(path)]
    judge_ok = gated_ok = scored = flipped = 0
    for r in rows:
        g = gold[r["id"]]
        if r.get("correct") is None:
            continue
        scored += 1
        before = r.get("correct_judge_only", r["correct"])
        judged = g["gold_value"] is None and g["type"] not in ("unanswerable", "refusal", "out_of_corpus", "injection")
        after = before
        if judged and before and M.figure_gate(g["gold_answer"], str(r.get("answer", ""))) is False:
            after = False
            flipped += 1
        r["correct_judge_only"], r["correct"] = before, after
        judge_ok += bool(before)
        gated_ok += bool(after)
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return judge_ok, gated_ok, scored, flipped


def main() -> None:
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "gold.jsonl")}
    paths = sys.argv[1:] or sorted(glob.glob(str(HERE / "results" / "baselines" / "*.jsonl")))
    lines = ["| run | judge only | with figure gate | flipped to wrong |", "|---|---|---|---|"]
    for p in paths:
        j, g, n, f = rescore_file(p, gold)
        lines.append(f"| {Path(p).stem} | {j}/{n} ({j / max(1, n):.0%}) | {g}/{n} ({g / max(1, n):.0%}) | {f} |")
    out = Path(paths[0]).parent / "rescored.md" if paths else None
    if out:
        out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
