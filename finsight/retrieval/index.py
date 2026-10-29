"""Hybrid index: metadata filter -> BM25 + dense -> reciprocal rank fusion ->
cross-encoder rerank.

Build:  python -m finsight.retrieval.index --strategy table
        python -m finsight.retrieval.index --strategy table --no-dense   (BM25 only)

Each ablation config (dense only / +BM25 / +rerank / +filters) is a set of
flags on `HybridIndex.search`, so eval/retrieval_eval.py can sweep them.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from finsight import tracing
from finsight.config import INDEX_DIR, SECTIONS_DIR, settings
from finsight.retrieval.chunking import Chunk, chunk_section
from finsight.retrieval.rerank import get_reranker

# Boilerplate items that are mostly "incorporated by reference" - skipping
# them saves embedding quota without losing answerable content.
DEFAULT_ITEMS = {"1", "1A", "1C", "2", "3", "5", "7", "7A", "8", "9A"}

_STOP = set("the a an of and or to in for on by with as is are was were be been this that from at it its which".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9\.\-&]*", text.lower())
    return [t.rstrip(".") for t in toks if t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: dict = field(default_factory=dict)  # {"bm25": rank, "dense": rank, "rerank": score}


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None, embedder=None):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self._doc_ids = np.array([c.doc_id for c in chunks])

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, strategy: str = "table", size: int = 1800, overlap: int = 200, dense: bool = True,
              items: set[str] | None = DEFAULT_ITEMS, name: str = "default") -> "HybridIndex":
        secs = [json.loads(l) for l in open(SECTIONS_DIR / "sections.jsonl")]
        if items:
            secs = [s for s in secs if s["item"] in items]
        chunks = [c for s in secs for c in chunk_section(s, strategy, size, overlap)]
        print(f"{len(secs)} sections -> {len(chunks)} chunks ({strategy}, size={size}, overlap={overlap})")
        vectors, embedder = None, None
        if dense:
            from finsight.retrieval.embed import Embedder

            embedder = Embedder()
            vectors = embedder.embed([c.text for c in chunks], progress=True)
        out = INDEX_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "chunks.jsonl", "w") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        if vectors is not None:
            np.save(out / "vectors.npy", vectors)
        (out / "meta.json").write_text(json.dumps({"strategy": strategy, "size": size, "overlap": overlap,
                                                     "dense": dense, "n_chunks": len(chunks)}))
        return cls(chunks, vectors, embedder)

    @classmethod
    def load(cls, name: str = "default") -> "HybridIndex":
        d = INDEX_DIR / name
        chunks = [Chunk(**json.loads(l)) for l in open(d / "chunks.jsonl")]
        vectors, embedder = None, None
        if (d / "vectors.npy").exists() and settings().embed_backend == "api":
            from finsight.retrieval.embed import Embedder

            vectors, embedder = np.load(d / "vectors.npy"), Embedder()
        return cls(chunks, vectors, embedder)

    # --------------------------------------------------------------- search
    def _mask(self, ticker: str | list[str] | None, fiscal_year: int | list[int] | None, item: str | None) -> np.ndarray:
        m = np.ones(len(self.chunks), dtype=bool)
        tickers = [ticker] if isinstance(ticker, str) else ticker
        years = [fiscal_year] if isinstance(fiscal_year, int) else fiscal_year
        if tickers:
            m &= np.array([c.ticker in tickers for c in self.chunks])
        if years:
            m &= np.array([c.fiscal_year in years for c in self.chunks])
        if item:
            m &= np.array([c.item == item.upper() for c in self.chunks])
        if not m.any():  # a bad filter must not return nothing; fall back to unfiltered
            return np.ones(len(self.chunks), dtype=bool)
        return m

    def search(self, query: str, k: int = 5, *, ticker=None, fiscal_year=None, item: str | None = None,
               use_bm25: bool = True, use_dense: bool = True, use_rerank: bool = True,
               use_filters: bool = True, pool: int = 50, rrf_k: int = 60) -> list[Hit]:
        with tracing.span("search", "retrieval", input={"q": query, "ticker": ticker, "fy": fiscal_year,
                                                         "item": item, "k": k}) as sp:
            mask = self._mask(ticker, fiscal_year, item) if use_filters else np.ones(len(self.chunks), bool)
            idx = np.flatnonzero(mask)
            ranks: dict[int, dict] = {}
            if use_bm25:
                scores = self.bm25.get_scores(tokenize(query))[idx]
                for r, j in enumerate(idx[np.argsort(-scores)[:pool]]):
                    ranks.setdefault(int(j), {})["bm25"] = r
            if use_dense and self.vectors is not None and self.embedder is not None:
                qv = self.embedder.embed([query])[0]
                sims = self.vectors[idx] @ qv
                for r, j in enumerate(idx[np.argsort(-sims)[:pool]]):
                    ranks.setdefault(int(j), {})["dense"] = r
            # reciprocal rank fusion
            fused = sorted(((sum(1.0 / (rrf_k + r) for r in src.values()), j, src) for j, src in ranks.items()),
                           reverse=True)[: max(pool // 2, k)]
            hits = [Hit(self.chunks[j], s, dict(src)) for s, j, src in fused]
            if use_rerank and hits:
                rr = get_reranker(settings().reranker, settings().reranker_model)
                if rr.name != "none":
                    scores = rr.rerank(query, [h.chunk.text for h in hits])
                    for h, s in zip(hits, scores):
                        h.sources["rerank"] = s
                        h.score = s
                    hits.sort(key=lambda h: -h.score)
            hits = hits[:k]
            sp.output = [h.chunk.chunk_id for h in hits]
        return hits


@lru_cache(maxsize=1)
def default_index() -> HybridIndex:
    return HybridIndex.load("default")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="table", choices=["fixed", "recursive", "table"])
    ap.add_argument("--size", type=int, default=1800)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--all-items", action="store_true")
    ap.add_argument("--name", default="default")
    a = ap.parse_args()
    HybridIndex.build(a.strategy, a.size, a.overlap, dense=not a.no_dense,
                      items=None if a.all_items else DEFAULT_ITEMS, name=a.name)
