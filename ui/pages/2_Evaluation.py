"""Latest evaluation, retrieval ablation and safety results."""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2] / "eval" / "results"
st.set_page_config(page_title="FinSight evaluation", layout="wide")
st.title("Evaluation")

runs = sorted([p for p in ROOT.glob("*/summary.md")], key=lambda p: p.stat().st_mtime, reverse=True)
if runs:
    pick = st.selectbox("Eval run", [p.parent.name for p in runs])
    st.markdown((ROOT / pick / "summary.md").read_text())
else:
    st.info("No eval runs yet: `python -m eval.run_eval`")

for title, path in [("Baselines (Week 1)", ROOT / "baselines" / "baseline_table.md"),
                    ("Retrieval ablation (Week 2)", ROOT / "retrieval_default.md"),
                    ("Responsible AI (Week 4)", ROOT / "safety" / "safety_report.md")]:
    st.subheader(title)
    st.markdown(path.read_text() if path.exists() else "_not run yet_")
