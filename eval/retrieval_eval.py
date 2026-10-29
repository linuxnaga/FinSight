"""Week-2 retrieval ablation: recall@k measured separately from answer quality.

    python -m eval.retrieval_eval                 # ablation on the default index
    python -m eval.retrieval_eval --index fixed   # compare a different chunking index

Build alternative chunkings first, e.g.
    python -m finsight.retrieval.index --strategy fixed --name fixed
    python -m finsight.retrieval.index --strategy recursive --name recursive
"""
from __future__ import annotations

import argparse
from pathlib import Path

from finsight.retrieval.index import HybridIndex
from eval.run_eval import load_gold

CONFIGS = {
    "bm25 only": dict(use_dense=False, use_rerank=False, use_filters=False),
    "bm25 + filters": dict(use_dense=False, use_rerank=False),
    "bm25 + filters + item prior": dict(use_dense=False, use_rerank=False, use_item_prior=True),
    "dense only": dict(use_bm25=False, use_rerank=False, use_filters=False),
    "dense + filters": dict(use_bm25=False, use_rerank=False),
    "hybrid (RRF) + filters": dict(use_rerank=False),
    "hybrid + filters + item prior": dict(use_rerank=False, use_item_prior=True),
    "hybrid + filters + rerank": dict(),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="default")
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    a = ap.parse_args()
    ix = HybridIndex.load(a.index)
    items = [i for i in load_gold() if i["evidence_doc_ids"]]
    has_dense = ix.vectors is not None
    from finsight.config import settings
    from finsight.retrieval.rerank import get_reranker
    has_rerank = get_reranker(settings().reranker, settings().reranker_model).name != "none"
    rows = []
    for name, cfg in CONFIGS.items():
        if not has_dense and cfg.get("use_dense", True):
            continue
        if cfg.get("use_rerank", True) and not has_rerank:
            continue
        rec = {k: [] for k in a.ks}
        for it in items:
            hits = ix.search(it["question"], k=max(a.ks), ticker=it["ticker"], fiscal_year=it["fiscal_year"], **cfg)
            docs = [h.chunk.doc_id for h in hits]
            for k in a.ks:
                rec[k].append(sum(g in docs[:k] for g in it["evidence_doc_ids"]) / len(it["evidence_doc_ids"]))
        rows.append((name, {k: sum(v) / len(v) for k, v in rec.items()}))
        print(f"{name:28s} " + "  ".join(f"R@{k}={v:.2f}" for k, v in rows[-1][1].items()))
    md = ["| config | " + " | ".join(f"recall@{k}" for k in a.ks) + " |", "|---" * (len(a.ks) + 1) + "|"]
    md += [f"| {n} | " + " | ".join(f"{v[k]:.2f}" for k in a.ks) + " |" for n, v in rows]
    out = Path(__file__).parent / "results" / f"retrieval_{a.index}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"Index `{a.index}`, {len(items)} questions with gold evidence. Recall is at section level.\n\n"
                   + "\n".join(md) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
