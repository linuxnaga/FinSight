"""FinSight chat UI. Talks to the API (so rate limiting and caching apply).

    streamlit run ui/app.py
"""
import os
import uuid

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="FinSight", page_icon="📊", layout="wide")
st.title("FinSight: 10-K analyst")
st.caption("Answers come from SEC 10-K filings (FY2020–FY2023) of 20 companies. Draft output: "
           "verify before use. No investment advice.")

if "session" not in st.session_state:
    st.session_state.session = uuid.uuid4().hex[:8]
    st.session_state.turns = []

with st.sidebar:
    st.subheader("Try")
    examples = ["What was Nike's revenue in FY2023?",
                "What was Adobe's operating margin in FY2023?",
                "Which had higher revenue in FY2022, Coca-Cola or PepsiCo?",
                "What supply chain risks did PepsiCo highlight in FY2022?",
                "and the year before?",
                "Should I buy Boeing stock?"]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    if st.button("New conversation"):
        st.session_state.session = uuid.uuid4().hex[:8]
        st.session_state.turns = []
    st.caption(f"session `{st.session_state.session}`")


def render(turn: dict) -> None:
    r = turn["result"]
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        if r.get("refused"):
            st.warning(r["answer"])
        else:
            st.markdown(r["answer"])
        if r.get("value") is not None:
            st.metric("headline value", f"{r['value']:,.2f} {r.get('unit') or ''}")
        cols = st.columns(5)
        cols[0].caption(f"route: **{r.get('route')}**")
        cols[1].caption(f"tool calls: **{r.get('tool_calls')}**")
        cols[2].caption(f"latency: **{r.get('latency_s')}s**")
        cols[3].caption(f"tokens: **{r.get('tokens')}**" + (" (cached)" if r.get("cached") else ""))
        cols[4].caption(f"trace: `{r.get('trace_id')}`")
        if r.get("citations"):
            with st.expander(f"Sources ({len(r['citations'])})"):
                for c in r["citations"]:
                    link = f" [open filing]({c['url']})" if c.get("url") else ""
                    st.markdown(f"**[{c['id']}]** `{c['source']}`{link}")
                    st.caption(c["snippet"])
        if r.get("plan"):
            with st.expander("Plan & checks"):
                st.json({"plan": r["plan"], "critiques": r.get("critiques"), "failures": r.get("failures")})
        if r.get("injection_flags"):
            st.info(f"{r['injection_flags']} retrieved passage(s) contained instruction-like text and were neutralised.")
        st.caption(r.get("disclaimer", ""))


for t in st.session_state.turns:
    render(t)

q = st.chat_input("Ask about a 10-K…") or st.session_state.pop("pending", None)
if q:
    with st.spinner("Planning, retrieving, checking…"):
        try:
            resp = httpx.post(f"{API}/ask", json={"question": q, "session_id": st.session_state.session}, timeout=180)
            if resp.status_code in (429, 503):
                result = {"answer": resp.json()["detail"], "refused": "unavailable"}
            else:
                resp.raise_for_status()
                result = resp.json()
        except Exception as e:
            result = {"answer": f"API error: {e}", "refused": "error"}
    st.session_state.turns.append({"q": q, "result": result})
    st.rerun()
