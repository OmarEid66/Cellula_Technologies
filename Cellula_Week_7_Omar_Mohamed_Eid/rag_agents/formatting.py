"""Small helpers that render evidence for prompts and for people."""
from __future__ import annotations

import re

from .schemas import Evidence


def location_label(e: Evidence) -> str:
    if e.page is not None:
        return f"page {e.page}"
    if e.section:
        return f'section "{e.section}"'
    return "document"


def evidence_id(index: int) -> str:
    """Evidence ids are positional (E1, E2, ...) so they stay stable while evidence is only appended."""
    return f"E{index + 1}"


def parse_evidence_id(value: str, total: int) -> int | None:
    m = re.fullmatch(r"\s*\[?E(\d+)\]?\s*", str(value), flags=re.IGNORECASE)
    if not m:
        return None
    idx = int(m.group(1)) - 1
    return idx if 0 <= idx < total else None


def format_evidence_block(evidence: list[Evidence], max_chars: int = 2500) -> str:
    blocks = []
    for i, e in enumerate(evidence):
        text = e.content if len(e.content) <= max_chars else e.content[:max_chars] + " ...[truncated]"
        blocks.append(
            f"[{evidence_id(i)}] source={e.source} | {location_label(e)} | type={e.chunk_type} "
            f"| score={e.score:.2f}\n{text}"
        )
    return "\n\n".join(blocks) if blocks else "(no evidence)"


def format_history(history: list[dict] | None, max_messages: int = 6) -> str:
    if not history:
        return "(no previous messages)"
    lines = []
    for m in history[-max_messages:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = str(m.get("content", ""))
        lines.append(f"{role}: {content[:600]}")
    return "\n".join(lines)
