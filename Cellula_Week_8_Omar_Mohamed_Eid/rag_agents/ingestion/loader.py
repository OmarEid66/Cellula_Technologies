"""LOADER: read PDF / DOCX / TXT files into pages (text + raw tables) with source metadata."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .ocr import IMAGE_EXTENSIONS, ocr_image

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"} | IMAGE_EXTENSIONS


@dataclass
class LoadedPage:
    text: str
    page: int | None = None  # real page number (PDF) or None when the format has no pages
    section: str | None = None  # nearest heading (DOCX)
    tables: list[list[list[str]]] = field(default_factory=list)  # each table = rows of cells


@dataclass
class LoadedFile:
    source: str  # file name used for citations
    path: str
    doc_type: str  # pdf | docx | txt
    pages: list[LoadedPage]
    warnings: list[str] = field(default_factory=list)
    ocr_confidence: float | None = None  # mean OCR word confidence (images only)


def load_file(
    path: str | os.PathLike,
    source_name: str | None = None,
    ocr_languages: str = "eng",
    ocr_min_confidence: int = 50,
) -> LoadedFile:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    source = source_name or p.name
    if ext == ".pdf":
        return _load_pdf(p, source)
    if ext == ".docx":
        return _load_docx(p, source)
    if ext in IMAGE_EXTENSIONS:
        return _load_image(p, source, ocr_languages, ocr_min_confidence)
    return _load_text(p, source)


def _cell(value) -> str:
    return "" if value is None else str(value).strip()


def _load_pdf(p: Path, source: str) -> LoadedFile:
    from pypdf import PdfReader

    warnings: list[str] = []
    reader = PdfReader(str(p))
    pages: list[LoadedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - corrupt page
            warnings.append(f"page {i}: text extraction failed ({exc})")
            text = ""
        pages.append(LoadedPage(text=text, page=i))

    # tables (pdfplumber); failures only cost the table chunks, never the whole file
    try:
        import pdfplumber

        with pdfplumber.open(str(p)) as pdf:
            for i, plumber_page in enumerate(pdf.pages):
                if i >= len(pages):
                    break
                try:
                    for table in plumber_page.extract_tables():
                        rows = [[_cell(c) for c in row] for row in table if row and any(_cell(c) for c in row)]
                        if len(rows) >= 2:
                            pages[i].tables.append(rows)
                except Exception as exc:  # pragma: no cover
                    warnings.append(f"page {i + 1}: table extraction failed ({exc})")
    except Exception as exc:  # pragma: no cover
        warnings.append(f"table extraction unavailable ({exc})")

    if not any(pg.text.strip() for pg in pages):
        warnings.append("no extractable text found (scanned PDF? upload its pages as images so they are OCR-ed)")
    return LoadedFile(source=source, path=str(p), doc_type="pdf", pages=pages, warnings=warnings)


def _load_docx(p: Path, source: str) -> LoadedFile:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(p))
    pages: list[LoadedPage] = []
    current = LoadedPage(text="", section=None)
    buffer: list[str] = []

    def flush():
        nonlocal current, buffer
        current.text = "\n".join(buffer)
        if current.text.strip() or current.tables:
            pages.append(current)
        buffer = []

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            style = (para.style.name or "") if para.style is not None else ""
            if style.lower().startswith("heading") or style.lower() == "title":
                flush()
                current = LoadedPage(text="", section=para.text.strip() or None)
                if para.text.strip():
                    buffer.append(para.text.strip())
            elif para.text.strip():
                buffer.append(para.text)
        elif tag == "tbl":
            table = Table(child, doc)
            rows = [[_cell(c.text) for c in row.cells] for row in table.rows]
            rows = [r for r in rows if any(r)]
            if len(rows) >= 2:
                current.tables.append(rows)
    flush()
    warnings = [] if pages else ["document has no extractable text"]
    return LoadedFile(source=source, path=str(p), doc_type="docx", pages=pages, warnings=warnings)


def _load_image(p: Path, source: str, languages: str, min_confidence: int) -> LoadedFile:
    """OCR an image; its text becomes a single unpaged document so it is chunked and embedded like any other."""
    result = ocr_image(p, languages=languages, min_confidence=min_confidence)
    warnings = list(result.warnings)
    return LoadedFile(
        source=source,
        path=str(p),
        doc_type="image",
        pages=[LoadedPage(text=result.text, page=None)] if result.has_text else [],
        warnings=warnings,
        ocr_confidence=result.confidence,
    )


def _load_text(p: Path, source: str) -> LoadedFile:
    raw = p.read_bytes()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    parts = text.split("\f")  # form feed = page break, otherwise a single unpaged document
    if len(parts) > 1:
        pages = [LoadedPage(text=t, page=i) for i, t in enumerate(parts, start=1)]
    else:
        pages = [LoadedPage(text=text, page=None)]
    warnings = [] if text.strip() else ["file is empty"]
    return LoadedFile(source=source, path=str(p), doc_type="txt", pages=pages, warnings=warnings)
