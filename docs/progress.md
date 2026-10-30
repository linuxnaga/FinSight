# Progress log

Each entry records what changed, why, and the measured effect. The report's "consistent progress" and "what we did
to compensate for a small model" sections can be drawn directly from here.

## v0.3: first run against the course endpoint, evaluation fixes

**Endpoint facts** (found with `make probe`): the course endpoint serves `Qwen/Qwen2.5-7B-Instruct-AWQ`, not Llama.
It supports `json_schema` constrained decoding and has no `/embeddings` route, so dense retrieval uses the local CPU
model. Gateway errors arrive as HTTP 200 with `{"error": ...}`; `LLM.chat` now reports them clearly.

**Two evaluation bugs found in the first baseline run** (fixed before any results were reported):
| Bug | Symptom | Fix |
|---|---|---|
| Oracle mode passed the first 7,000 characters of the gold *section*, but gold sections are ~160k characters (all of Item 8), so the evidence was usually missing | "oracle" scored barely above closed-book (22.9% vs 14.3%); answers said "not provided in the given excerpts" | gold items now carry `evidence_text` (FinanceBench evidence pages; reported XBRL figures), and oracle mode uses exactly that |
| The rounding rule matched **any** number in the answer, at any scale and either sign | AES ROA answered 6.19% scored correct vs gold −0.02 (the "/ 2" in the working → 2% → 0.02) | rounding only for the headline value, same sign; scales limited to ×10³ and percent↔fraction; years and small integers in prose ignored; regression tests use the exact failing answer |

**Rate limiting.** The rerun showed oracle at 34.3% vs 14.3% closed-book (the oracle fix works), but about half of all
requests had failed with HTTP 429 and were scored as wrong answers. The endpoint rate-limits without advertising
a limit, so the client now paces requests (`LLM_MAX_RPM`, default 20/min) and retries 429/5xx with exponential
backoff (`LLM_MAX_RETRIES`, default 6). Baselines and the eval exclude calls that still fail (including judge
failures) from accuracy and report them in their own column. In a live check (4 calls while another run was
saturating the endpoint), five 429s were retried and all 4 calls succeeded.

**LLM judge too lenient.** With rate limiting fixed, closed-book few-shot scored 37%, high for a 7B model with no
documents. Reading the judged-correct answers showed the 7B judge accepting wrong figures and even contradictions:
"inventory turnover is not meaningful" for gold 9.5x, J&J growth "6%" for gold 1.3%, AmEx tax rate 22.6% → 21.8%
for gold 24.6% → 21.6%, "margins fluctuated" for gold "consistent". Fixes:
* **figure gate** (`metrics.figure_gate`): if the gold states key figures, the answer must state at least one at the
  gold's precision ("21.6%" ± 0.05, "23%" ± 0.5). It can only fail answers, never rescue them. Abstains when the gold
  has no figures.
* stricter judge rubric: a wrong key figure caps the score at 3; contradicting the gold or "not available" scores 1–2.
* `make rescore` re-applies the gate to saved results without model calls; `make audit` exports 30 judged answers
  for a teammate to label, and `python -m eval.judge_audit score ...` reports judge-vs-human agreement.

Effect on the first gated runs (few-shot): closed-book 37% → **23%**, oracle 83% → **63%**.

Re-scored with the strict metric: few-shot oracle numeric 2/11 → 1/11; other runs unchanged. The no-model floor
stays at **26/33**: it only ever scored headline values, which the strict rule still allows.

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
* `numeric_match` accepts the gold answer's own rounding (FinanceBench writes ROA −1.53% as "-0.02") and percent vs
  fraction, but only for the **headline value** with the same sign. See v0.3 for why.

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
