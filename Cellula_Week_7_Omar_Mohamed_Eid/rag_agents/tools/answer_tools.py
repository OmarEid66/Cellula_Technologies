"""Answer Agent tools: citation formatter, source formatter, response formatter."""
from __future__ import annotations

import re

from ..schemas import Citation, Evidence


def citation_formatter(evidence: list[Evidence], used_indices: list[int]) -> tuple[list[Citation], dict[int, int]]:
    """Group evidence by (document, page/section) and number the groups 1..n in order of first use.

    Returns the citations and a map evidence-index -> citation number.
    """
    citations: list[Citation] = []
    key_to_number: dict[tuple, int] = {}
    index_to_number: dict[int, int] = {}
    for idx in used_indices:
        ev = evidence[idx]
        key = (ev.source, ev.page, None if ev.page is not None else ev.section)
        if key not in key_to_number:
            number = len(citations) + 1
            key_to_number[key] = number
            citations.append(Citation(number=number, source=ev.source, page=ev.page, section=ev.section if ev.page is None else None))
        number = key_to_number[key]
        citations[number - 1].chunk_ids.append(ev.chunk_id)
        index_to_number[idx] = number
    return citations, index_to_number


def source_formatter(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["**Sources**"] + [f"- [{c.number}] {c.label()}" for c in citations]
    return "\n".join(lines)


def response_formatter(text: str, valid_numbers: set[int]) -> str:
    """Tidy the model's answer: drop citation markers that point nowhere, de-duplicate repeats, fix table spacing."""

    def _marker(m: re.Match) -> str:
        nums = [int(n) for n in re.findall(r"\d+", m.group(0))]
        keep = [n for n in nums if n in valid_numbers]
        return "".join(f"[{n}]" for n in dict.fromkeys(keep))

    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", _marker, text)
    text = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", text)  # [1][1] -> [1]
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"([^\n|])\n(\|.*\|\n\|[\s:|-]+\|)", r"\1\n\n\2", text)  # blank line before a markdown table
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
