Index `local`, 88 questions with gold evidence. Recall is at section level.

| config | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| bm25 only | 0.06 | 0.23 | 0.36 |
| bm25 + filters | 0.42 | 0.78 | 0.89 |
| bm25 + filters + item prior | 0.47 | 0.73 | 0.81 |
| dense only | 0.09 | 0.60 | 0.79 |
| dense + filters | 0.51 | 0.90 | 0.95 |
| hybrid (RRF) + filters | 0.48 | 0.92 | 0.94 |
| hybrid + filters + item prior | 0.49 | 0.82 | 0.86 |
| hybrid + filters + rerank | 0.44 | 0.88 | 0.97 |
