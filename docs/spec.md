# FinSight: Task specification (Week 1 deliverable)

## What the copilot does
Answers analyst questions about the annual reports (Form 10-K) of 20 US-listed companies for fiscal years
2020–2023, with every claim cited to a filing section or an XBRL fact.

| In scope | Example |
|---|---|
| Reported figures | "What was Nike's revenue in FY2023?" |
| Derived metrics | "Adobe's operating margin in FY2023?", "Pfizer revenue change FY22→FY23?" |
| Peer / year comparisons | "Coca-Cola vs PepsiCo revenue in FY2022" |
| Narrative questions | "What supply-chain risks did PepsiCo highlight in FY2022?" |
| Follow-ups | "…and the year before?" (conversation state persists per session) |

Companies: AMD, Boeing, American Express, PepsiCo, 3M, Adobe, Amcor, Best Buy, Verizon, Corning, CVS Health,
General Mills, Nike, AES, Amazon, American Water Works, Coca-Cola, Johnson & Johnson, Lockheed Martin, Pfizer.

Fiscal-year convention: the calendar year in which the fiscal year **ends** (Nike FY2023 = Jun 2022–May 2023;
Best Buy FY2023 = Feb 2022–Jan 2023). This matches how the companies label their own years.

## What it must refuse
| Category | Behaviour |
|---|---|
| Investment advice (buy/sell/hold, "is it a good time to…") | Refuse; offer to summarise reported facts instead |
| Price targets / share-price predictions | Refuse |
| Personal data about individuals | Refuse |
| Companies or years outside the corpus | Say so and list what is covered |
| Questions unrelated to filings | Decline briefly |
| Instructions embedded in documents | Ignore; neutralise; never follow |

Refusals for the first four are **rule-based** (no LLM call), so they're deterministic and cost no tokens.

## What a correct answer looks like
* **Numeric questions:** headline value within ±1% of gold (scale-insensitive: millions vs billions), correct
  company, fiscal year and unit.
* **Narrative questions:** LLM-as-judge score ≥ 4/5 against the gold answer (rubric in `eval/metrics.py`).
* **Grounding:** at least one citation pointing to a gold evidence section or an XBRL fact; no numbers that
  don't appear in the evidence (checked by the critic).
* **Unanswerable:** says the information isn't in the filings (no invented figure).

## Human-in-the-loop boundary
FinSight produces **drafts for an analyst**. It never: makes recommendations, sends anything, or exports a report
without the "verify against source" label. Any consequential use (investment decision, published research, client
communication) requires a human to check the cited sources. The UI shows every source and the full trace.

## Evaluation set
`eval/gold.jsonl`: 100 questions, frozen at tag `v0.1-gold-frozen` before any agent code is evaluated.

| Source | n | Gold |
|---|---|---|
| FinanceBench (open-source subset) on our companies/years | 74 | expert-written answers + evidence pages, mapped to our section ids |
| XBRL-derived lookups / ratios / comparisons | 16 | exact values from SEC XBRL |
| Hand-written safety cases | 12 | expected refusal / "not found" |
