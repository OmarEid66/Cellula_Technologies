"""CLEANING: normalise extracted text before chunking."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\f\v]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if not _PAGE_NUMBER_LINE.match(ln)]
    text = "\n".join(lines)
    # single newlines inside a paragraph -> space; blank lines stay as paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def remove_repeated_lines(page_texts: list[str], min_pages: int = 4, ratio: float = 0.6) -> list[str]:
    """Drop running headers / footers: short lines at the top or bottom of most pages."""
    if len(page_texts) < min_pages:
        return page_texts
    counter: Counter[str] = Counter()
    for text in page_texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        edge = set(lines[:2] + lines[-2:])
        for ln in edge:
            if len(ln) <= 100:
                counter[ln] += 1
    repeated = {ln for ln, n in counter.items() if n >= max(3, ratio * len(page_texts))}
    if not repeated:
        return page_texts
    cleaned = []
    for text in page_texts:
        cleaned.append("\n".join(ln for ln in text.splitlines() if ln.strip() not in repeated))
    return cleaned
