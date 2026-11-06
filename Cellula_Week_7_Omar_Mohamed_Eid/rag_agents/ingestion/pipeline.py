"""Document preparation pipeline: LOADER -> PARSER -> CLEANING -> CHUNKING -> EMBEDDINGS -> VECTOR DB."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..vectorstore import KnowledgeBase
from .chunking import chunk_documents
from .loader import load_file
from .parser import parse_file


@dataclass
class IngestReport:
    source: str
    pages: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and (self.text_chunks + self.table_chunks) > 0


def ingest_file(
    path: str | Path, kb: KnowledgeBase, settings: Settings, source_name: str | None = None
) -> IngestReport:
    name = source_name or Path(path).name
    report = IngestReport(source=name)
    try:
        loaded = load_file(path, source_name=name)
        report.pages = len(loaded.pages)
        report.warnings.extend(loaded.warnings)
        docs = parse_file(loaded)
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap, settings.max_table_chars)
        report.text_chunks = sum(1 for c in chunks if c.metadata["chunk_type"] == "text")
        report.table_chunks = sum(1 for c in chunks if c.metadata["chunk_type"] == "table")
        if not chunks:
            report.error = "no text could be extracted from this file"
            return report
        kb.delete_source(name)  # re-uploading a file replaces its old chunks
        kb.add_documents(chunks)  # embeddings are computed here and stored in the vector DB
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    return report


def ingest_files(
    paths: list[str | Path],
    kb: KnowledgeBase,
    settings: Settings,
    on_progress: Callable[[IngestReport], None] | None = None,
) -> list[IngestReport]:
    reports = []
    for p in paths:
        r = ingest_file(p, kb, settings)
        reports.append(r)
        if on_progress:
            on_progress(r)
    return reports
