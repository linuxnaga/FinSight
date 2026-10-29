"""HTTP API in front of the agent: rate limiting, response cache, traces.

    uvicorn finsight.api:app --port 8000
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from finsight.config import CACHE_DB, DATA_DIR, settings
from finsight.tracing import load_traces

app = FastAPI(title="FinSight", version="0.1.0")
_agent = None
_agent_lock = threading.Lock()


def agent():
    global _agent
    with _agent_lock:
        if _agent is None:
            from finsight.agent.graph import Agent

            try:
                _agent = Agent()
            except RuntimeError as e:  # endpoint not configured
                raise HTTPException(503, str(e)) from e
    return _agent


# ------------------------------------------------------------ rate limit
class RateLimiter:
    """Sliding window per client: protects the team's token quota from a
    runaway script or a demo-day refresh storm."""

    def __init__(self, per_min: int):
        self.per_min = per_min
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                return False, int(60 - (now - q[0])) + 1
            q.append(now)
            return True, 0


limiter = RateLimiter(settings().rate_limit_per_min)


# ----------------------------------------------------------------- cache
class Cache:
    def __init__(self, ttl_s: int = 24 * 3600):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE_DB, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, ts REAL, v TEXT)")
        self.ttl, self.lock = ttl_s, threading.Lock()

    @staticmethod
    def key(q: str) -> str:
        norm = re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", q.lower())).strip()
        return hashlib.sha1(norm.encode()).hexdigest()

    def get(self, q: str) -> dict | None:
        row = self.db.execute("SELECT ts, v FROM c WHERE k=?", (self.key(q),)).fetchone()
        if row and time.time() - row[0] < self.ttl:
            return json.loads(row[1])
        return None

    def put(self, q: str, v: dict) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?, ?, ?)", (self.key(q), time.time(), json.dumps(v)))
            self.db.commit()


cache = Cache()


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    session_id: str | None = Field(None, max_length=64)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": settings().llm_model}


@app.post("/ask")
def ask(body: AskIn, request: Request) -> dict:
    client = request.headers.get("x-forwarded-for", request.client.host if request.client else "anon")
    ok, retry = limiter.allow(client)
    if not ok:
        raise HTTPException(429, f"rate limit: {limiter.per_min} questions/minute. Retry in {retry}s",
                            headers={"Retry-After": str(retry)})
    a = agent()
    # Only first turns are cacheable; follow-ups depend on conversation state.
    first_turn = not body.session_id or not a.store.history(body.session_id, 1)
    if first_turn and (hit := cache.get(body.question)):
        return {**hit, "cached": True, "session": body.session_id or hit.get("session")}
    result = a.run(body.question, session=body.session_id)
    if first_turn and not any(f in ("internal_error", "budget_exceeded") for f in result["failures"]):
        cache.put(body.question, result)
    return {**result, "cached": False}


@app.get("/traces")
def traces(limit: int = 50) -> list[dict]:
    return load_traces(min(limit, 500))


@app.get("/traces/{trace_id}")
def trace(trace_id: str) -> dict:
    for t in load_traces(1000):
        if t["trace_id"] == trace_id:
            return t
    raise HTTPException(404, "trace not found")
