"""Minimal tracer. Every query gets one trace; every LLM call, tool call and
agent node is a span. Traces are written as JSONL to data/traces/ (always) and
mirrored to Langfuse when LANGFUSE_* keys are set.

Spans carry a `failure` tag when something goes wrong, which is what the
failure-taxonomy report aggregates over.
"""
from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from finsight.config import TRACE_DIR


@dataclass
class Span:
    name: str
    kind: str  # "llm" | "tool" | "node" | "retrieval"
    start: float
    end: float = 0.0
    input: Any = None
    output: Any = None
    error: str | None = None
    failure: str | None = None  # taxonomy tag, see eval/failure_taxonomy.py
    meta: dict = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str
    question: str
    start: float
    spans: list[Span] = field(default_factory=list)
    end: float = 0.0
    answer: Any = None
    tokens_prompt: int = 0
    tokens_completion: int = 0

    @property
    def tokens(self) -> int:
        return self.tokens_prompt + self.tokens_completion

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens"] = self.tokens
        d["latency_s"] = round(self.end - self.start, 3) if self.end else None
        return d


_current: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("trace", default=None)


def current() -> Trace | None:
    return _current.get()


@contextmanager
def trace(question: str) -> Iterator[Trace]:
    t = Trace(trace_id=uuid.uuid4().hex[:12], question=question, start=time.time())
    token = _current.set(t)
    try:
        yield t
    finally:
        t.end = time.time()
        _current.reset(token)
        _persist(t)


@contextmanager
def span(name: str, kind: str, input: Any = None, **meta) -> Iterator[Span]:
    s = Span(name=name, kind=kind, start=time.time(), input=_safe(input), meta=meta)
    t = current()
    if t is not None:
        t.spans.append(s)
    try:
        yield s
    except Exception as e:  # recorded, then re-raised
        s.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        s.end = time.time()
        s.output = _safe(s.output)


def add_usage(prompt_tokens: int, completion_tokens: int) -> None:
    t = current()
    if t is not None:
        t.tokens_prompt += prompt_tokens or 0
        t.tokens_completion += completion_tokens or 0


def _safe(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except TypeError:
        if hasattr(x, "model_dump"):
            return x.model_dump()
        return repr(x)


def _persist(t: Trace) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    day = time.strftime("%Y-%m-%d", time.localtime(t.start))
    with open(TRACE_DIR / f"{day}.jsonl", "a") as f:
        f.write(json.dumps(t.to_dict(), default=repr) + "\n")
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        _to_langfuse(t)


def _to_langfuse(t: Trace) -> None:
    """Best-effort export; a tracing outage must never break a user query."""
    try:
        from langfuse import Langfuse  # type: ignore

        lf = Langfuse()
        root = lf.trace(id=t.trace_id, name="finsight-query", input=t.question, output=t.answer,
                        metadata={"tokens": t.tokens})
        for s in t.spans:
            kw = dict(name=s.name, input=s.input, output=s.output,
                      metadata={**s.meta, "kind": s.kind, "failure": s.failure, "error": s.error})
            if s.kind == "llm":
                root.generation(**kw)
            else:
                root.span(**kw)
        lf.flush()
    except Exception:
        pass


def load_traces(limit: int = 200) -> list[dict]:
    if not TRACE_DIR.exists():
        return []
    rows: list[dict] = []
    for p in sorted(TRACE_DIR.glob("*.jsonl"), reverse=True):
        for line in reversed(p.read_text().splitlines()):
            rows.append(json.loads(line))
            if len(rows) >= limit:
                return rows
    return rows
