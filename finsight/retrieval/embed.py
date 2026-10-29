"""Embeddings, cached on disk by content hash so a re-index or a chunking
experiment only pays for chunks it hasn't seen.

Backends
  api    the course endpoint (counts against the team's token quota)
  local  sentence-transformers on CPU (`pip install .[rerank]`), e.g.
         BAAI/bge-small-en-v1.5: no quota, ~10 min for the full corpus on a laptop
"""
from __future__ import annotations

import hashlib
import sqlite3

import numpy as np

from finsight.config import DATA_DIR, settings

# bge models expect this instruction on queries (not on passages)
_QUERY_PREFIX = {"bge": "Represent this sentence for searching relevant passages: "}


class Embedder:
    def __init__(self, client=None, model: str | None = None, batch: int = 64, backend: str | None = None):
        s = settings()
        self.backend = backend or s.embed_backend
        self.model = model or s.embed_model
        self.batch = batch
        self.client = client
        self._st = None
        if self.backend == "local":
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st = SentenceTransformer(self.model, device="cpu")
        elif client is None:
            from openai import OpenAI

            if not s.embed_base_url:
                raise RuntimeError("EMBED_BASE_URL/LLM_BASE_URL is not set (or use EMBED_BACKEND=local|none)")
            self.client = OpenAI(base_url=s.embed_base_url, api_key=s.embed_api_key, timeout=120, max_retries=3)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(DATA_DIR / "embeddings.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.backend}:{self.model}\x00{text}".encode()).hexdigest()

    def _compute(self, texts: list[str]) -> list[np.ndarray]:
        if self._st is not None:
            return list(self._st.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False))
        resp = self.client.embeddings.create(model=self.model, input=[t[:8000] for t in texts])
        return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]

    def embed_query(self, query: str) -> np.ndarray:
        prefix = next((p for k, p in _QUERY_PREFIX.items() if k in self.model.lower()), "")
        return self.embed([prefix + query])[0]

    def embed(self, texts: list[str], progress: bool = False) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        have = {}
        for i in range(0, len(keys), 900):
            q = keys[i : i + 900]
            for k, v in self.db.execute(f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(q))})", q):
                have[k] = np.frombuffer(v, dtype=np.float32)
        todo = [i for i, k in enumerate(keys) if k not in have]
        for n, start in enumerate(range(0, len(todo), self.batch)):
            idx = todo[start : start + self.batch]
            for i, v in zip(idx, self._compute([texts[i] for i in idx])):
                v = np.asarray(v, dtype=np.float32)
                have[keys[i]] = v
                self.db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?)", (keys[i], v.tobytes()))
            self.db.commit()
            if progress and n % 20 == 0:
                print(f"  embedded {min(start + self.batch, len(todo))}/{len(todo)}", flush=True)
        m = np.vstack([have[k] for k in keys]) if keys else np.zeros((0, 1), dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)
