"""Split ingested document text into page-aware, overlapping chunks for retrieval."""

import re
from dataclasses import dataclass

PAGE_MARKER = re.compile(r"^\[Page (\d+)\]\s*$", re.M)


@dataclass
class Chunk:
    index: int
    page: int
    text: str


def split_pages(text: str) -> list[tuple[int, str]]:
    """'[Page N]' markers come from app.ingest; text without markers is treated as page 1."""
    marks = list(PAGE_MARKER.finditer(text))
    if not marks:
        return [(1, text.strip())] if text.strip() else []
    pages = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        if body:
            pages.append((int(m.group(1)), body))
    return pages


def chunk_text(text: str, size: int = 1200, overlap: int = 200, min_size: int = 80) -> list[Chunk]:
    """Pack whole lines into ~`size`-char chunks that never cross a page boundary.

    Consecutive chunks on a page share ~`overlap` chars so a sentence split across
    chunks is still retrievable. Fragments shorter than `min_size` (page numbers,
    headers) are dropped unless they are the only content on the page.
    """
    chunks: list[Chunk] = []
    for page, body in split_pages(text):
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        current: list[str] = []
        page_chunks: list[str] = []
        for line in lines:
            if current and len(" ".join(current)) + len(line) + 1 > size:
                page_chunks.append(" ".join(current))
                # carry trailing lines (up to `overlap` chars) into the next chunk
                carry, n = [], 0
                for prev in reversed(current):
                    if n + len(prev) > overlap:
                        break
                    carry.insert(0, prev)
                    n += len(prev) + 1
                current = carry
            current.append(line)
        if current:
            tail = " ".join(current)
            if not page_chunks or len(tail) >= min_size:
                page_chunks.append(tail)
        for t in page_chunks:
            if len(t) >= min_size or len(page_chunks) == 1:
                chunks.append(Chunk(index=len(chunks), page=page, text=t))
    return chunks
