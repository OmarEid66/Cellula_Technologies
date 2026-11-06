"""CHUNKING: split parsed documents into retrieval-sized chunks, keeping metadata for citations."""
from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _split_table(content: str, max_chars: int) -> list[str]:
    """Split an oversized markdown table by rows, repeating the header in every part."""
    if len(content) <= max_chars:
        return [content]
    lines = content.splitlines()
    # lines[0] = title line, lines[1:3] = header + separator, rest = body rows
    title, header, body = lines[0], lines[1:3], lines[3:]
    parts, current = [], []
    size = 0
    for row in body:
        if current and size + len(row) > max_chars:
            parts.append("\n".join([title, *header, *current]))
            current, size = [], 0
        current.append(row)
        size += len(row) + 1
    if current:
        parts.append("\n".join([title, *header, *current]))
    return parts or [content]


def chunk_documents(
    docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150, max_table_chars: int = 3500
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    counters: dict[str, int] = {}
    for doc in docs:
        meta = dict(doc.metadata)
        if meta.get("chunk_type") == "table":
            pieces = _split_table(doc.page_content, max_table_chars)
        else:
            pieces = splitter.split_text(doc.page_content)
        for piece in pieces:
            if not piece.strip():
                continue
            key = f"{meta['source']}|{meta.get('page', 0)}|{meta['chunk_type']}"
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            digest = hashlib.sha1(f"{key}|{idx}|{piece[:80]}".encode()).hexdigest()[:12]
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={**meta, "chunk_index": idx, "chunk_id": f"{digest}"},
                )
            )
    return chunks
