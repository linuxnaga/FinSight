"""Turn a 10-K HTML document into one record per Item (Item 1, 1A, 7, 8 ...).

Each Item is a "document" in the corpus sense: 20 companies x 4 years x ~20
items ~= 1,600 documents, under the 2,000 cap. Chunking happens later.

Tables are flattened to pipe-delimited rows so retrieval and the model can
still see row/column structure.
"""
from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "Reserved / Selected Financial Data",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

_ITEM_RE = re.compile(
    r"^\s*item[\s\xa0]*(1a|1b|1c|7a|9a|9b|9c|1[0-6]|[1-9])\s*[\.:\-—–]?\s*(.{0,200})$",
    re.I | re.M,
)


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # inline-XBRL header block is hidden metadata, not filing text
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in ("$", ")", "%")]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(soup.new_string("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n"))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]):
        block.insert_after("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    # page furniture: running "Table of Contents" links and bare page numbers
    text = re.sub(r"(?im)^\s*(table of contents|\d{1,3}|.{0,40}form 10-k\s*\d{1,3})\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_items(text: str) -> dict[str, str]:
    """Split the 10-K text into Items. The table of contents also contains
    'Item 7.' lines, so for each Item we keep the longest span: the ToC entry
    is a one-liner, the real section is pages long."""
    matches = list(_ITEM_RE.finditer(text))
    best: dict[str, str] = {}
    for i, m in enumerate(matches):
        key = m.group(1).upper()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        if len(body) > len(best.get(key, "")):
            best[key] = body
    # Drop near-empty stubs ("Item 4. Not applicable.") only if truly empty
    return {k: v for k, v in best.items() if len(v) > 40}


def sections_for_filing(html: str | bytes, *, ticker: str, company: str, fiscal_year: int, url: str) -> list[dict]:
    items = split_items(html_to_text(html))
    out = []
    for key, body in items.items():
        out.append({
            "doc_id": f"{ticker}_{fiscal_year}_10K_item{key}",
            "ticker": ticker,
            "company": company,
            "fiscal_year": fiscal_year,
            "form": "10-K",
            "item": key,
            "title": ITEM_TITLES.get(key, ""),
            "text": body,
            "url": url,
        })
    return out
