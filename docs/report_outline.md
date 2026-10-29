# Technical report outline (12 pages)

Each section names the artifact in the repo that supplies its evidence.

1. **Problem & scope** (1 p): `docs/spec.md`
2. **Evaluation design** (1 p): gold-set construction, metrics, judge rubric, freeze tag. `eval/build_gold.py`, `eval/metrics.py`
3. **Week-1 baselines and how the 8B model breaks** (1.5 p)
   - Table: 3 strategies × closed/oracle/naive-RAG. `eval/results/baselines/baseline_table.md`
   - Breakage log with examples: malformed JSON, unit/scale confusion, wrong fiscal year, arithmetic slips,
     confident answers with no evidence, ignoring "not found". `eval/results/baselines/*.jsonl`
4. **Retrieval** (2 p)
   - Chunking experiment: fixed vs recursive vs table-aware (`--name fixed|recursive|default`)
   - Ablation: BM25 / dense / hybrid RRF / +rerank / +metadata filters, recall@1/5/10. `eval/results/retrieval_*.md`
   - Finding to discuss: metadata filters alone moved BM25 recall@5 from 0.22 → 0.78
5. **Agent architecture** (2 p): guard → route → plan → execute → answer → critic loop; tools; state; limits.
   `finsight/agent/graph.py`, `finsight/tools/registry.py`
6. **Reliability engineering** (1 p): JSON repair + retry, schema-validated tool args, enum-constrained tickers
   and metrics, "$N" step references, loop guard, step cap, token budget, timeouts, fallbacks.
   Evidence: JSON repairs/failures counters in eval summaries; `failures` field in traces.
7. **What we did to compensate for a small model** (1.5 p, graded section). For each, a before/after number:
   | Technique | Measured by |
   |---|---|
   | Arithmetic moved to `calculate` | computed-question accuracy, baseline vs agent |
   | Numbers from XBRL instead of table reading | lookup accuracy, baseline vs agent |
   | Rule-based refusal + entity extraction | refusal accuracy; LLM calls saved |
   | Deterministic fast path for single-metric lookups | tokens/query, latency |
   | Metadata filters + hybrid + rerank | recall@5 |
   | Critic numeric-grounding check | hallucination count before/after (`MAX_CRITIC_REVISIONS=0` vs 2) |
   | Short single-purpose prompts | JSON failure rate |
8. **End-to-end results** (1 p): accuracy by type, completion rate, tool calls/task, tokens & latency per query.
   `eval/results/<run>/summary.md`
9. **Failure taxonomy** (0.5 p): retrieval miss / reasoning error / tool-call error / hallucination / refusal error,
   with 2 traced examples each (trace ids from `results.jsonl`)
10. **Responsible AI** (1 p): prompt-injection results (regex screen + live), PII masking, refusal behaviour,
    human-in-the-loop boundary. `eval/results/safety/safety_report.md`
11. **Deployment** (0.5 p): FastAPI + Streamlit, Docker Compose, rate limit, cache, tracing
12. **Limitations & next steps** (0.5 p)
