"""Download the corpus and build data/sections/sections.jsonl.

    python -m finsight.data.ingest                # all companies
    python -m finsight.data.ingest --tickers NKE  # subset
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from finsight.config import COMPANIES, FISCAL_YEARS, SECTIONS_DIR
from finsight.data import edgar
from finsight.data.sections import sections_for_filing


def main(tickers: list[str]) -> None:
    manifest = [m for m in edgar.load_manifest() if m["ticker"] not in tickers]
    for t in tickers:
        cik, name = COMPANIES[t]
        rows = edgar.download_10ks(t, cik, FISCAL_YEARS)
        print(f"{t:5s} 10-Ks for FY {sorted(r['fiscal_year'] for r in rows)}")
        manifest.extend(rows)
    edgar.save_manifest(manifest)

    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SECTIONS_DIR / "sections.jsonl"
    n, per_item = 0, Counter()
    with open(out, "w") as f:
        for m in sorted(manifest, key=lambda r: (r["ticker"], r["fiscal_year"])):
            html = Path(m["path"]).read_bytes()
            secs = sections_for_filing(html, ticker=m["ticker"], company=COMPANIES[m["ticker"]][1],
                                       fiscal_year=m["fiscal_year"], url=m["url"])
            if len(secs) < 8:
                print(f"  WARN {m['ticker']} FY{m['fiscal_year']}: only {len(secs)} items parsed")
            for s in secs:
                f.write(json.dumps(s) + "\n")
                per_item[s["item"]] += 1
                n += 1
    print(f"\n{n} section documents from {len(manifest)} filings -> {out}")
    print("per item:", dict(sorted(per_item.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=list(COMPANIES))
    a = ap.parse_args()
    main(a.tickers)
