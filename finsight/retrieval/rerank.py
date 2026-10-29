"""Reranking. The cross-encoder runs on CPU (no GPU needed for ~30 pairs per
query). If sentence-transformers isn't installed the identity reranker is used
and the ablation table simply shows "no rerank"."""
from __future__ import annotations

from functools import lru_cache


class IdentityReranker:
    name = "none"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(len(texts) - i) for i in range(len(texts))]


class CrossEncoderReranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model = CrossEncoder(model, max_length=512)
        self.name = f"cross-encoder:{model}"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16)]


@lru_cache(maxsize=2)
def get_reranker(kind: str, model: str):
    if kind == "cross-encoder":
        try:
            return CrossEncoderReranker(model)
        except ImportError:
            print("[rerank] sentence-transformers not installed -> using no reranker (pip install .[rerank])")
    return IdentityReranker()
