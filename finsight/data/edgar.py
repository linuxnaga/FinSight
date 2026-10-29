"""SEC EDGAR client: 10-K filing list, primary documents and XBRL company facts.

SEC fair-access rules: identify yourself in the User-Agent and stay under
10 requests/second. Everything is cached on disk so the demo never depends
on EDGAR being reachable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from finsight.config import RAW_DIR, XBRL_DIR, fiscal_year_of, settings

_last_call = 0.0


def _get(url: str) -> httpx.Response:
    global _last_call
    wait = 0.15 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    headers = {"User-Agent": settings().sec_user_agent, "Accept-Encoding": "gzip, deflate"}
    for attempt in range(4):
        _last_call = time.time()
        r = httpx.get(url, headers=headers, timeout=60, follow_redirects=True)
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    r.raise_for_status()
    return r


def company_facts(ticker: str, cik: int, refresh: bool = False) -> dict:
    path = XBRL_DIR / f"{ticker}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    XBRL_DIR.mkdir(parents=True, exist_ok=True)
    data = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
    path.write_text(json.dumps(data))
    return data


def ten_k_filings(cik: int) -> list[dict]:
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    # "recent" holds the last ~1,000 filings; heavy filers (e.g. Verizon's note
    # issuances) push older 10-Ks into the paginated "files" list.
    pages = [sub["filings"]["recent"]]
    for extra in sub["filings"].get("files", []):
        pages.append(_get(f"https://data.sec.gov/submissions/{extra['name']}").json())
    rows = []
    for page in pages:
        for i, form in enumerate(page["form"]):
            if form == "10-K":
                rows.append({
                    "accession": page["accessionNumber"][i],
                    "primary_doc": page["primaryDocument"][i],
                    "report_date": page["reportDate"][i],
                    "filing_date": page["filingDate"][i],
                })
    return rows


def filing_url(cik: int, accession: str, primary_doc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{primary_doc}"


def download_10ks(ticker: str, cik: int, years: list[int]) -> list[dict]:
    """Download the 10-K primary documents for the requested fiscal years.
    Returns manifest rows (one per filing)."""
    company_facts(ticker, cik)  # cache XBRL alongside the documents
    out_dir = RAW_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for f in ten_k_filings(cik):
        fy = fiscal_year_of(f["report_date"])
        if fy not in years:
            continue
        path = out_dir / f"{fy}.htm"
        url = filing_url(cik, f["accession"], f["primary_doc"])
        if not path.exists():
            path.write_bytes(_get(url).content)
        manifest.append({"ticker": ticker, "cik": cik, "fiscal_year": fy, "path": str(path), "url": url, **f})
    return manifest


def load_manifest() -> list[dict]:
    p = RAW_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else []


def save_manifest(rows: list[dict]) -> Path:
    p = RAW_DIR / "manifest.json"
    p.write_text(json.dumps(rows, indent=1))
    return p
