from docx import Document as DocxDocument
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from rag_agents.ingestion import ingest_file
from rag_agents.ingestion.cleaning import clean_text, remove_repeated_lines
from rag_agents.ingestion.loader import load_file
from rag_agents.ingestion.parser import parse_file, rows_to_markdown
from rag_agents.tools.analysis import parse_markdown_tables


def make_pdf(path):
    styles = getSampleStyleSheet()
    table = Table([["Model", "Accuracy", "F1"], ["CNN-A", "92.1", "0.90"], ["CNN-B", "95.4", "0.94"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    doc.build(
        [
            Paragraph("Page one introduces the benchmark and the models under test.", styles["Normal"]),
            PageBreak(),
            Paragraph("Results on page two are summarised below.", styles["Normal"]),
            table,
        ]
    )


def test_pdf_pages_and_tables(tmp_path):
    p = tmp_path / "report.pdf"
    make_pdf(p)
    loaded = load_file(p)
    assert [pg.page for pg in loaded.pages] == [1, 2]
    docs = parse_file(loaded)
    tables = [d for d in docs if d.metadata["chunk_type"] == "table"]
    assert len(tables) == 1 and tables[0].metadata["page"] == 2
    parsed = parse_markdown_tables(tables[0].page_content)
    assert parsed[0]["columns"] == ["Model", "Accuracy", "F1"]
    assert parsed[0]["rows"][1]["Accuracy"] == "95.4"


def test_docx_sections_and_tables(tmp_path):
    p = tmp_path / "spec.docx"
    d = DocxDocument()
    d.add_heading("Overview", level=1)
    d.add_paragraph("This document describes the system.")
    d.add_heading("Metrics", level=1)
    t = d.add_table(rows=2, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Metric", "Value"
    t.rows[1].cells[0].text, t.rows[1].cells[1].text = "Latency", "12 ms"
    d.save(p)
    loaded = load_file(p)
    assert [pg.section for pg in loaded.pages] == ["Overview", "Metrics"]
    docs = parse_file(loaded)
    assert any(x.metadata["chunk_type"] == "table" and x.metadata["section"] == "Metrics" for x in docs)


def test_txt_and_unsupported(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    assert load_file(p).pages[0].text == "hello world"
    bad = tmp_path / "a.xyz"
    bad.write_text("x")
    try:
        load_file(bad)
        assert False
    except ValueError:
        pass


def test_cleaning():
    raw = "Intro-\nduction to   RAG\n\n12\n\nsecond\nparagraph"
    out = clean_text(raw)
    assert "Introduction to RAG" in out and "\n12\n" not in out and "second paragraph" in out
    pages = [f"Company Confidential\nbody {i}\nPage {i}" for i in range(5)]
    cleaned = remove_repeated_lines(pages)
    assert all("Company Confidential" not in c for c in cleaned)


def test_rows_to_markdown_escapes_pipes():
    md = rows_to_markdown([["a", "b"], ["x|y", "2"]])
    assert "x\\|y" in md


def test_ingest_replaces_old_chunks_and_reports(tmp_path, kb, settings):
    p = tmp_path / "doc.txt"
    p.write_text("alpha beta gamma. " * 80, encoding="utf-8")
    r1 = ingest_file(p, kb, settings)
    n1 = kb.count()
    r2 = ingest_file(p, kb, settings)
    assert r1.ok and r2.ok and kb.count() == n1  # no duplicates after re-upload
    empty = tmp_path / "empty.txt"
    empty.write_text("   ")
    assert not ingest_file(empty, kb, settings).ok
    assert not ingest_file(tmp_path / "missing.pdf", kb, settings).ok
