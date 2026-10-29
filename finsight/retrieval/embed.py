"""Embeddings via the course endpoint, cached on disk by content hash so a
re-index or a chunking experiment only pays for chunks it hasn't seen."""
from __future__ import annotations

import hashlib
import sqlite3

import numpy as np

from finsight.config import DATA_DIR, settings


class Embedder:
    def __init__(self, client=None, model: str | None = None, batch: int = 64):
        s = settings()
        if client is None:
            from openai import OpenAI

            if not s.embed_base_url:
                raise RuntimeError("EMBED_BASE_URL/LLM_BASE_URL is not set (or use EMBED_BACKEND=none)")
            client = OpenAI(base_url=s.embed_base_url, api_key=s.embed_api_key, timeout=120, max_retries=3)
        self.client, self.model, self.batch = client, model or s.embed_model, batch
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(DATA_DIR / "embeddings.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}\x00{text}".encode()).hexdigest()

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
            resp = self.client.embeddings.create(model=self.model, input=[texts[i][:8000] for i in idx])
            for i, d in zip(idx, resp.data):
                v = np.asarray(d.embedding, dtype=np.float32)
                have[keys[i]] = v
                self.db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?)", (keys[i], v.tobytes()))
            self.db.commit()
            if progress and n % 20 == 0:
                print(f"  embedded {min(start + self.batch, len(todo))}/{len(todo)}")
        m = np.vstack([have[k] for k in keys]) if keys else np.zeros((0, 1), dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)
