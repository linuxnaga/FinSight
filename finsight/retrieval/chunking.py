"""Chunking strategies compared in Week 2.

  fixed      - fixed-size character windows with overlap (the naive baseline)
  recursive  - paragraph-aware packing with overlap
  table      - like recursive, but [TABLE] blocks are kept whole (or split by
               rows with the header row repeated) so numbers stay next to
               their row/column labels

Every chunk is prefixed with a context header ("Nike FY2023 10-K, Item 7
Management's Discussion...") which helps both BM25 and a weak generator know
where a passage comes from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ticker: str
    company: str
    fiscal_year: int
    item: str
    title: str
    text: str
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


def header(sec: dict) -> str:
    return f"{sec['company']} ({sec['ticker']}) FY{sec['fiscal_year']} 10-K, Item {sec['item']} {sec['title']}"


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def _pack(blocks: list[str], size: int, overlap: int) -> list[str]:
    chunks, cur = [], ""
    for b in blocks:
        if len(b) > size:  # oversized paragraph -> window it
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_fixed(b, size, overlap))
            continue
        if len(cur) + len(b) + 2 > size and cur:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n\n" + b) if tail else b
        else:
            cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        chunks.append(cur)
    return chunks


def _split_table(table: str, size: int) -> list[str]:
    lines = table.strip().splitlines()
    inner = [l for l in lines if l not in ("[TABLE]", "[/TABLE]")]
    if len(table) <= size or len(inner) < 3:
        return [table.strip()]
    head = inner[0]
    parts, cur = [], [head]
    for row in inner[1:]:
        if sum(len(r) for r in cur) + len(row) > size and len(cur) > 1:
            parts.append("[TABLE]\n" + "\n".join(cur) + "\n[/TABLE]")
            cur = [head]
        cur.append(row)
    if len(cur) > 1:
        parts.append("[TABLE]\n" + "\n".join(cur) + "\n[/TABLE]")
    return parts


def split_text(text: str, strategy: str = "table", size: int = 1800, overlap: int = 200) -> list[str]:
    if strategy == "fixed":
        return _fixed(text, size, overlap)
    if strategy == "recursive":
        return _pack([p.strip() for p in text.split("\n\n") if p.strip()], size, overlap)
    if strategy == "table":
        blocks: list[str] = []
        for piece in re.split(r"(\[TABLE\].*?\[/TABLE\])", text, flags=re.S):
            piece = piece.strip()
            if not piece:
                continue
            if piece.startswith("[TABLE]"):
                blocks.extend(_split_table(piece, size))
            else:
                blocks.extend(p.strip() for p in piece.split("\n\n") if p.strip())
        # tables pack with neighbouring paragraphs when they fit, but are never
        # windowed mid-row (oversized tables were already split by rows above)
        out, cur = [], ""
        for b in blocks:
            if len(cur) + len(b) + 2 > size and cur:
                out.append(cur)
                tail = "" if b.startswith("[TABLE]") or cur.endswith("[/TABLE]") else cur[-overlap:]
                cur = f"{tail}\n\n{b}" if tail else b
            else:
                cur = f"{cur}\n\n{b}" if cur else b
            if len(cur) > size * 1.5 and not cur.startswith("[TABLE]"):
                out.extend(_fixed(cur, size, overlap))
                cur = ""
        if cur:
            out.append(cur)
        return out
    raise ValueError(f"unknown chunking strategy {strategy}")


def chunk_section(sec: dict, strategy: str = "table", size: int = 1800, overlap: int = 200) -> list[Chunk]:
    h = header(sec)
    return [
        Chunk(chunk_id=f"{sec['doc_id']}#{i}", doc_id=sec["doc_id"], ticker=sec["ticker"], company=sec["company"],
              fiscal_year=sec["fiscal_year"], item=sec["item"], title=sec["title"], text=f"{h}\n{body}",
              url=sec["url"])
        for i, body in enumerate(split_text(sec["text"], strategy, size, overlap))
        if len(body.strip()) > 30
    ]
