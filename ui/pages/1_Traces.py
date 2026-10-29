"""Trace dashboard: every real query execution, its spans, tokens and failures."""
import os
from collections import Counter

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="FinSight traces", layout="wide")
st.title("Traces")

try:
    traces = httpx.get(f"{API}/traces", params={"limit": 300}, timeout=30).json()
except Exception as e:
    st.error(f"API unreachable: {e}")
    st.stop()
if not traces:
    st.info("No traces yet. Ask a question first.")
    st.stop()

rows = []
for t in traces:
    a = t.get("answer") or {}
    fails = [s["failure"] for s in t["spans"] if s.get("failure")] + list(a.get("failures") or [])
    rows.append({"trace": t["trace_id"], "question": t["question"][:80], "route": a.get("route"),
                 "latency_s": t.get("latency_s"), "tokens": t.get("tokens"),
                 "tool_calls": sum(s["kind"] == "tool" for s in t["spans"]),
                 "llm_calls": sum(s["kind"] == "llm" for s in t["spans"]),
                 "refused": a.get("refused"), "failures": ", ".join(sorted(set(fails)))})
df = pd.DataFrame(rows)

c = st.columns(4)
c[0].metric("queries", len(df))
c[1].metric("p50 latency (s)", f"{df.latency_s.median():.1f}")
c[2].metric("avg tokens", f"{df.tokens.mean():.0f}")
c[3].metric("avg tool calls", f"{df.tool_calls.mean():.2f}")

fail_counts = Counter(f.split(":")[0] for fs in df.failures for f in fs.split(", ") if f)
if fail_counts:
    st.subheader("Failure tags")
    st.bar_chart(pd.Series(fail_counts, name="count"))

st.subheader("Recent queries")
st.dataframe(df, use_container_width=True, hide_index=True)

tid = st.selectbox("Inspect trace", df.trace)
t = next(x for x in traces if x["trace_id"] == tid)
st.write(f"**{t['question']}**")
t0 = t["start"]
for s in t["spans"]:
    label = f"{s['kind']:9s} {s['name']}  (+{s['start'] - t0:.2f}s, {s['end'] - s['start']:.2f}s)"
    if s.get("failure") or s.get("error"):
        label += f"  ⚠ {s.get('failure') or s.get('error')}"
    with st.expander(label):
        st.json({"input": s.get("input"), "output": s.get("output"), "meta": s.get("meta")})
