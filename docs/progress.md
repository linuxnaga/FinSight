# Progress log

Each entry records what changed, why, and the measured effect. The report's "consistent progress" and "what we did
to compensate for a small model" sections can be drawn directly from here.

## v0.5: second agent run, judge stability, narrative retrieval

**Agent rerun** (`20261030-143138`, same 30 questions): **63% (19/30)**, up from 60%. Task completion **100%**
(was 93%), computed **9/9**, hallucinations 0, 429 retries 3. Narrative still 4/15.

**The judge, not the agent, caused both "regressions".** Two Boeing answers were word-for-word identical in both
runs, yet the judge passed them once and failed them the next time, after a one-line rubric change. The 7B judge
also kept failing Corning's 20.48% → 22.87% against gold "20% → 23%" despite the rounding rule. Fixes:
* **verdict cache** keyed by rubric version: identical answers get identical verdicts, so run-to-run differences
  measure the agent
* **figure note**: when the deterministic check confirms the gold's figures at its precision, the judge is told so
  and grades the conclusion only. Live re-judging: Corning and Boeing production rates now pass. Boeing cyclicality
  is still failed ("does not explicitly state…" when it does), a remaining judge error for the human audit to quantify

**Why narrative questions failed: the agent's searches, not the index.** Agent search recall@5 was 0.12, against
0.92 for the same index offline. Traces showed:
| Bug | Effect | Fix |
|---|---|---|
| planner passed `item="10-K"`, `"10K"`, `"301"`; nothing matched, and the fallback dropped *all* filters | Boeing questions retrieved Adobe, Verizon and General Mills filings | item values validated (junk means no preference); filters relax least-important-first and never drop the company |
| a planner-chosen item was a hard filter | all 5 passages from one (often wrong) section, e.g. Item 7 when the answer is in Item 1A | the item is a soft preference: unfiltered results plus up to 2 from that item |
| paraphrased queries ("acquisition major deal") | missed the right passages | LLM-planned questions are led by a search with the question as asked; `$N` step references skip it |

Offline check with the agent's own entity extraction: the lead search finds a gold section for **48/55 narrative
questions (87%)**, with **0** results from another company. 65 tests.

## v0.4: first agent evaluation (Qwen2.5-7B, 30-question subset)

### Week-1 baselines (40-question subset, gated scoring, 0 endpoint errors)
| Strategy | Closed-book | Naive RAG | Oracle evidence |
|---|---|---|---|
| few-shot JSON | 23% (numeric 1/11) | 34% (2/11) | 63% (8/11) |
| CoT + self-consistency | 17% (0/11) | 26% (2/11) | 51% (9/11) |
| decomposition | 17% (0/11) | 26% (2/11) | 49% (8/11) |

The model knows almost no figures (closed-book numeric 0–1/11); evidence triples accuracy; naive retrieval recovers
little of that gap; reasoning prompts cost up to 5x the calls and don't beat structured few-shot on a 7B model.

### Agent (`make eval-quick`, run `20261030-110333`)
| Metric | Value |
|---|---|
| End-to-end accuracy | **60% (18/30)** |
| Computed / lookup | 8/9, 2/2 |
| Narrative | 4/15 |
| Refusal / injection / unanswerable | 2/2, 1/1, 1/1 |
| Task completion | 93% (2 budget overruns) |
| Avg tool calls / tokens per query | 2.2 / 3,818 |
| Latency p50 / p95 | 3.9 s / 28.8 s |
| JSON repairs / failures | **0 / 0** (`json_schema` constrained decoding) |

Different question subset from the baseline, so the comparison is indicative; the full 100-question run gives the exact figure.

### Fixes from reading the agent's failures
| Failure | Cause | Fix |
|---|---|---|
| GIS retention ratio 0.5489 vs gold 0.54 | the model filled its own `value`, and the agent kept it over the tool's correct 0.5403 | a tool-computed number always becomes the headline |
| Adobe "improving Free cashflow conversion?" → planner failed 4x | "cashflow" (one word) not recognised; one year given for a change question | normalise "cashflow"; change questions also compute the prior year |
| "Which debt securities…" / "Are there product categories >20% of revenue" | deterministic lookup fast path grabbed yes/no and "which" questions | those go to the router |
| Amcor acquisitions FY21–23: "Stopped: token budget exceeded" | 12k budget too small for 3-year narrative; overrun discarded the draft | budget 20k; an overrun returns the best answer so far, flagged |
| Corning tax rate 20.48% → 22.87% marked wrong vs gold "20% → 23%" | strict judge rubric treated rounding as a different figure | rubric: a figure that rounds to the gold figure matches |
| "retrieval recall@5 (agent searches) 0.00" | tools run in a worker thread without the trace context, so search spans were lost (also missing from the Traces page) | tool threads run in the caller's context |

Each has a regression test built from the live failure (62 tests).

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
