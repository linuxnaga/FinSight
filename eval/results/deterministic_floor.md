# Deterministic floor (no LLM)

- numeric questions answered exactly: **26/33**
- refusal / out-of-corpus handled by rules: **9/10**
- questions routed and planned with zero LLM calls: **66/100**

| type | correct (of scorable) |
|---|---|
| comparison | 0/2 |
| computed | 18/22 |
| injection | 2/2 |
| lookup | 8/9 |
| out_of_corpus | 3/4 |
| refusal | 4/4 |

Misses on numeric questions:

- `financebench_id_01319` got None vs gold 0.0 (plan: search_filings)
- `financebench_id_05718` got 389.0 vs gold 0.4 (plan: search_filings)
- `financebench_id_10130` got 64.8331 vs gold 63.86 (plan: search_filings)
- `financebench_id_02981` got None vs gold 10.3 (plan: search_filings)
- `financebench_id_03718` got None vs gold 0.4 (plan: search_filings)
- `peer_01` got None vs gold 43388.0 (plan: search_filings)
- `peer_02` got None vs gold 9142.0 (plan: search_filings)
