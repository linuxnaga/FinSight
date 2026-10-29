# FinSight: Financial-filings analyst agent

Capstone Track B: a multi-agent copilot built only on a hosted ≤8B model behind an OpenAI-compatible API.
It answers questions about 80 SEC 10-K filings (20 companies × FY2020–2023), with citations, exact numbers from
XBRL, a planner → executor → critic loop, and an evaluation harness.

```
Streamlit UI ──► FastAPI (rate limit + cache)
                    │
   guard (rules: refusal, PII, entities, follow-ups)
     → route (rules first, LLM if ambiguous)
     → plan  (fast path for single-metric lookups, else LLM planner with validated tool args)
     → execute (tools, "$N" step refs, fallbacks, loop guard, step cap, token budget)
     → answer (cited, evidence-only)
     → critic (number/citation checks + LLM groundedness for narrative) ──► rewrite / replan (≤2)
                    │ tools
   search_filings (BM25 + dense → RRF → cross-encoder, metadata filters)
   get_financial_fact (SEC XBRL)   calculate (safe eval)   compare_peers
                    │
   traces → data/traces/*.jsonl (+ Langfuse if configured) → Traces page
```

## Status

| Piece | State |
|---|---|
| Corpus: 80 10-Ks downloaded and split into 1,650 Item sections; XBRL facts for 20 companies | ✅ built (`make data`) |
| Eval set: 100 questions (74 FinanceBench + 16 XBRL-exact + 12 safety) | ✅ built; **team must review, then freeze** |
| BM25 index (26k chunks), retrieval ablation | ✅ runs offline |
| Dense embeddings, reranker | ⏳ needs course endpoint / `pip install .[rerank]` |
| Agent, tools, critic, memory, guardrails | ✅ 21 tests passing with a scripted model |
| API, UI, trace dashboard, Docker | ✅ runs locally |
| Baselines, full eval, live injection test | ⏳ need the endpoint key |

Measured so far, with no LLM involved:
- **Retrieval:** BM25 recall@5 is 0.22 without metadata filters and **0.78 with company/year filters**.
- **Injection screen:** the regex catches 5/10 planted injections; the other 5 rely on the prompt and critic, which is what the live test measures.
- **PII masking:** 4/4 cases masked.

## Setup

```bash
make setup                  # uv venv + install (Python 3.12)
cp .env.example .env        # add the course endpoint, key, model, and your SEC contact
make probe                  # which JSON mode does the endpoint support? set LLM_JSON_MODE
make data                   # ~5 min: EDGAR download + split (cached, safe to rerun)
make index                  # hybrid index (embeddings) - or `make index-bm25` without an endpoint
make test
make api                    # :8000
make ui                     # :8501 (chat, Traces, Evaluation pages)
# or: docker compose up --build
```

**Embedding quota:** the default index embeds 26k chunks (~10M tokens). If the team quota is smaller, use one of these:
- Raise `--size` to 3000.
- Restrict the index to Items 1A/7/8.
- Run `--no-dense` and report BM25 + filters + rerank (already 0.78 recall@5).

Embeddings are cached by content hash, so re-indexing is free.

## Week-by-week commands

| Week | Do | Command | Output |
|---|---|---|---|
| 1 | Review `eval/gold.jsonl`, then freeze it | `git tag v0.1-gold-frozen` | the gate |
| 1 | Baselines: 3 strategies × closed/oracle/naive-RAG | `make baseline` | `eval/results/baselines/` |
| 1 | Write the breakage log from the baseline outputs | n/a | `eval/results/baselines/breakage_log.md` |
| 2 | Chunking comparison | `python -m finsight.retrieval.index --strategy fixed --name fixed` (+ recursive) then `python -m eval.retrieval_eval --index fixed` | `eval/results/retrieval_*.md` |
| 2 | Hybrid + rerank ablation | `make retrieval` | same |
| 3 | Agent eval on a daily subset | `make eval-quick` | `eval/results/<run>/summary.md` |
| 3–4 | Full eval + failure taxonomy | `make eval` | same, plus traces |
| 4 | Injection / PII tests | `make safety` | `eval/results/safety/` |

Small-model ablations for the report: set `MAX_CRITIC_REVISIONS=0` (no critic), or `RERANKER=none`, and re-run `make eval-quick`.

## Team roles → code ownership

| Role | Owns |
|---|---|
| Domain spec & eval design | `docs/spec.md`, `eval/build_gold.py`, `eval/gold.jsonl`, judge rubric in `eval/metrics.py` |
| Prompt engineering | `finsight/agent/prompts.py`, `eval/baselines.py`, breakage log |
| RAG / retrieval | `finsight/data/`, `finsight/retrieval/`, `eval/retrieval_eval.py` |
| Agent orchestration & tools | `finsight/agent/graph.py`, `finsight/tools/`, `finsight/agent/memory.py` |
| Observability & safety | `finsight/tracing.py`, `finsight/agent/guard.py`, `eval/safety_eval.py`, failure taxonomy |
| Front-end & deployment | `finsight/api.py`, `ui/`, `Dockerfile`, `docker-compose.yml` |

Report skeleton: `docs/report_outline.md`.

## Notes
- **SEC:** set `SEC_USER_AGENT` to your team name and email (SEC fair-access policy). Downloads are throttled to under 10 requests/second.
- **Fiscal years:** labelled by the calendar year the fiscal year ends, which matches each company's own naming.
- **Evidence ids in the gold set:** FinanceBench evidence pages were mapped to our section ids automatically. Spot-check them during the review.
