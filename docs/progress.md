# Progress log

Each entry records what changed, why, and the measured effect. The report's "consistent progress" and "what we did
to compensate for a small model" sections can be drawn directly from here.

## v0.2: deterministic layer, data fixes, retrieval options

### Data correctness
| Issue found | Fix | Effect |
|---|---|---|
| J&J's 52/53-week fiscal years end in early January (FY2022 ends 2023-01-01), so FY2022 was labelled FY2023 and collided with the real FY2023 | `config.fiscal_year_of`: a period ending 1–7 Jan belongs to the prior year; used for both filings and XBRL | J&J filings re-downloaded; gold `xbrl_06` corrected 7.81 → **6.73** (FY2022 diluted EPS) |
| J&J's FY2023 filing carries a stray `0` R&D value for FY2021 under a different tag | Across candidate tags, prefer the value from the **original** 10-K | gold `xbrl_12` corrected 14.72% → **15.69%** |
| Companies report figures under tags the tool didn't know (Adobe R&D, Boeing interest, Verizon capex, PepsiCo/Lockheed receivables, J&J/Pfizer dividends, debt) | More tags; **derived** fallbacks (gross profit = revenue − COGS; total liabilities = L&E − equity; SG&A = S&M + G&A), labelled as derived in the source | XBRL coverage → **91%** (1,964/2,160 metric-years); remaining gaps are mostly things a company doesn't report (bank COGS, dividends at AMD/Amazon) |

### Deterministic layer (compensating for an 8B model)
| Change | Measured effect (`make floor`, no LLM at all) |
|---|---|
| `compute_ratio` tool: 30+ ratios with fixed formulas, prior-year averages, YoY growth | numeric gold answered exactly: 13 → **26 / 33** |
| Ratio/growth/peer-comparison fast paths (no LLM planner) | routed and planned with zero LLM calls: **66 / 100** questions |
| Metric detection ignores boilerplate ("statement of cash flows" no longer reads as *cash*); ratios ordered by first mention | fixed several misroutes (KO operating cash flow, PepsiCo EBITDA questions) |
| Entity fixes: "JnJ", default to latest FY when none is given (and say so), forward-looking questions also search the prior year's 10-K, FY2018–19 answered from comparatives | companies 85 → **88/88**, years 82 → **85/88** |
| Refusal rules cover "recommend buying", "stock picker", "will double" | injection-style gold questions refused by rules: 0 → **2/2** |

Remaining numeric misses need the model: 3-year averages and CAGR, a non-standard DPO definition, peer
differences that must be stated in prose, and one restructuring question that is answered from text.

### Scoring
* `numeric_match` accepts an answer that matches at the gold answer's own rounding (FinanceBench writes
  ROA −1.53% as "-0.02") and handles percent vs fraction. A test covers the false-positive edge.

### Retrieval
Section-level recall on the 88 gold questions with evidence. Dense = `BAAI/bge-small-en-v1.5` on CPU
(`make index-local`, 26,040 chunks, ~15 min, no quota). Reranker = `BAAI/bge-reranker-base` on CPU.

| Config | R@1 | R@5 | R@10 |
|---|---|---|---|
| BM25 only | 0.06 | 0.23 | 0.36 |
| BM25 + company/year filters | 0.42 | 0.78 | 0.89 |
| BM25 + filters + item prior | 0.47 | 0.73 | 0.81 |
| dense only | 0.09 | 0.60 | 0.79 |
| dense + filters | **0.51** | 0.90 | 0.95 |
| **hybrid (RRF) + filters**, the default | 0.48 | **0.92** | 0.94 |
| hybrid + filters + item prior | 0.49 | 0.82 | 0.86 |
| hybrid + filters + cross-encoder rerank | 0.44 | 0.88 | **0.97** |

Findings for the report:
* **Metadata filters are the biggest single win** (BM25 R@5 0.23 → 0.78). Small-model systems should extract
  entities with rules and filter, rather than hoping retrieval disambiguates "Nike FY2022" from "Nike FY2023".
* **Dense beats BM25** on this corpus (filtered R@5 0.90 vs 0.78). Hybrid adds a little at R@5.
* **The reranker is not free.** It trades top-5 precision for top-10 recall and costs ~1.6 s/query on CPU, so it's
  off by default (`RERANKER=none`). Worth re-testing with the endpoint's generator: if the answer step
  reads 10 passages, the reranker's R@10 gain may matter more.
* **The item prior hurts** at k ≥ 5, so it's off. We did not tune it on the eval set.

The index records which embedding model built it, so queries always use the same model. If that model can't be
loaded (e.g. the slim Docker image), search falls back to BM25 with a warning instead of failing.

### Engineering
* GitHub Actions CI: unit tests (corpus-dependent tests skip) plus a gold-set sanity check.
* 42 tests (was 21).

## v0.1: initial build
Corpus, gold set, hybrid index, agent loop, API/UI, Docker. See the README.
