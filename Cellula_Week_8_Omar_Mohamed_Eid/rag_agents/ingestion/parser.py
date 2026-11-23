"""PARSER: turn loaded pages into LangChain Documents (text pages + markdown tables)."""
from __future__ import annotations

from langchain_core.documents import Document

from .cleaning import clean_text, remove_repeated_lines
from .loader import LoadedFile


def rows_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    norm = [[(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    header, body = norm[0], norm[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def parse_file(loaded: LoadedFile) -> list[Document]:
    texts = remove_repeated_lines([pg.text for pg in loaded.pages])
    docs: list[Document] = []
    for pg, raw in zip(loaded.pages, texts):
        base_meta = {"source": loaded.source, "doc_type": loaded.doc_type}
        if pg.page is not None:
            base_meta["page"] = pg.page
        if pg.section:
            base_meta["section"] = pg.section

        text = clean_text(raw)
        if text:
            docs.append(Document(page_content=text, metadata={**base_meta, "chunk_type": "text"}))

        for table in pg.tables:
            where = f"page {pg.page}" if pg.page is not None else (f'section "{pg.section}"' if pg.section else "document")
            md = rows_to_markdown(table)
            docs.append(
                Document(
                    page_content=f"Table from {loaded.source}, {where}:\n{md}",
                    metadata={**base_meta, "chunk_type": "table"},
                )
            )
    return docs
