"""Report Generator Agent + its design toolkit (ReportLab). Everything runs offline with a scripted LLM."""
import io

import pytest
from langchain_core.messages import AIMessage
from pypdf import PdfReader

from fakes import ScriptedLLM
from rag_agents.agents.report import ReportAgent, build_default_report, format_material
from rag_agents.config import Settings
from rag_agents.schemas import ReportTurn
from rag_agents.tools.report_design import ReportDesigner, inline_markup, markdown_to_elements, parse_color

TURN = ReportTurn(
    question="What is the average CNN accuracy?",
    answer="The average is **92%** [1][2].\n\n| Paper | Accuracy |\n|---|---|\n| A | 92 |\n| B | 95 |\n\n- Paper C is lowest [3]",
    analysis="Accuracies are 92, 95 and 89.",
    calculations=["mean(92, 95, 89) = 92.0"],
    comparisons=["B > A > C"],
    sources=["paper_a_cnn.txt", "paper_b_cnn.txt"],
)


def pdf_text(path) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(str(path)).pages)


def pdf_fonts(path) -> set[str]:
    names = set()
    for page in PdfReader(str(path)).pages:
        for ref in page["/Resources"]["/Font"].values():
            names.add(str(ref.get_object()["/BaseFont"]).lstrip("/").split("+")[-1])
    return names


# ------------------------------------------------------------------ toolkit
def test_designer_renders_a_pdf_with_all_element_types(tmp_path):
    d = ReportDesigner("Demo")
    d.set_header_footer("Header text", "Footer text")
    d.add_title_block("Demo report", "sub", "Omar", "2026-01-01")
    d.add_heading("Section", 1, rule_below=True)
    d.add_paragraph("Hello **bold** *italic* {{#C0392B:red}} & <tags>", align="justify", underline=True, background_color="yellow")
    d.add_bullets(["one", "two"])
    d.add_bullets(["first", "second"], ordered=True)
    d.add_table(["Model", "Score"], [["A", "92%"], ["B", "95.5"]], caption="Scores")
    d.add_callout("Key point", "warning", title="Note")
    d.add_bar_chart(["A", "B"], [92, 95], "Accuracy")
    d.add_divider()
    d.add_spacer(10)
    d.add_page_break()
    d.add_markdown("# Heading\n\ntext\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    pages = d.render(tmp_path / "r.pdf")
    text = pdf_text(tmp_path / "r.pdf")
    assert pages == 2
    for expected in ["Demo report", "Header text", "Footer text", "Page 1", "Section", "Hello bold italic red & <tags>",
                     "Scores", "Key point", "Accuracy", "Heading"]:
        assert expected in text, expected


def test_design_tools_change_the_output(tmp_path):
    d = ReportDesigner()
    d.set_fonts(body_font="Times", heading_font="Courier")
    d.set_font_sizes(body=12, h1=22)
    d.set_colors(heading="#7B1E3A", accent="navy")
    d.set_page_layout("LETTER", "landscape", 15, 2)
    d.set_spacing(line_spacing=1.6, justify_body=True)
    d.add_heading("Title here", 1)
    d.add_paragraph("body text")
    d.render(tmp_path / "r.pdf")
    reader = PdfReader(str(tmp_path / "r.pdf"))
    assert (float(reader.pages[0].mediabox.width), float(reader.pages[0].mediabox.height)) == (792.0, 612.0)  # LETTER landscape
    assert {"Times-Roman", "Courier-Bold"} <= pdf_fonts(tmp_path / "r.pdf")


def test_invalid_design_values_raise_helpful_errors():
    d = ReportDesigner()
    with pytest.raises(ValueError, match="not available"):
        d.set_fonts("Comic Sans")
    with pytest.raises(ValueError, match="Unknown colour"):
        d.set_colors(accent="not-a-colour")
    with pytest.raises(ValueError, match="one per column"):
        d.add_table(["a", "b"], [["1", "2"]], column_widths_percent=[100])
    with pytest.raises(ValueError, match="same length"):
        d.add_bar_chart(["a"], [1, 2])
    with pytest.raises(ValueError, match="empty"):
        d.add_paragraph("   ")


def test_contrast_warning_and_review():
    d = ReportDesigner()
    msg = d.set_colors(page_background="#10222F", text="#222222")
    assert "low contrast" in msg
    d.add_heading("H", 1)
    review = d.review_report()
    assert "1 elements" in review and "heading" in review and "low contrast" in review


def test_shorthand_hex_colours_are_expanded():
    c = parse_color("#222")
    assert round(c.red * 255) == 0x22 and round(c.blue * 255) == 0x22


def test_inline_markup_escapes_and_converts():
    assert inline_markup("a & b < c") == "a &amp; b &lt; c"
    assert inline_markup("**x** *y* `z`") == '<b>x</b> <i>y</i> <font face="Courier">z</font>'


def test_malformed_markup_does_not_break_rendering(tmp_path):
    d = ReportDesigner()
    d.add_paragraph("bad nesting **a *b** c*")
    d.render(tmp_path / "r.pdf")
    assert "bad nesting" in pdf_text(tmp_path / "r.pdf")


def test_non_latin_text_still_renders(tmp_path):
    d = ReportDesigner()
    d.add_paragraph("Check mark ✓ and arrows → ≥ done")
    d.render(tmp_path / "r.pdf")
    assert "done" in pdf_text(tmp_path / "r.pdf")


def test_markdown_parser_blocks():
    els = markdown_to_elements("# T\n\npara one\ncontinues\n\n- a\n- b\n\n1. x\n2. y\n\n| h1 | h2 |\n|---|---|\n| 1 | 2 |\n\n---")
    assert [e["type"] for e in els] == ["heading", "paragraph", "bullets", "bullets", "table", "divider"]
    assert els[0]["level"] == 2  # '#' + default offset 1
    assert els[1]["text"] == "para one continues"
    assert els[2]["ordered"] is False and els[3]["ordered"] is True
    assert els[4]["headers"] == ["h1", "h2"] and els[4]["rows"] == [["1", "2"]]


def test_toolkit_exposes_the_design_controls():
    names = {t.name for t in ReportDesigner().tools()}
    assert {"set_fonts", "set_font_sizes", "set_colors", "set_spacing", "set_page_layout", "set_header_footer",
            "add_heading", "add_paragraph", "add_table", "add_bullets", "add_callout", "add_bar_chart",
            "add_markdown", "review_report"} <= names


# ------------------------------------------------------------------ agent
def scripted(batches):
    """LLM that issues the given tool-call batches one per step, then stops."""
    def step(messages):
        k = sum(1 for m in messages if getattr(m, "type", "") == "ai")
        if k < len(batches):
            return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{k}{i}"} for i, (n, a) in enumerate(batches[k])])
        return AIMessage(content="done")
    return ScriptedLLM(tool_step=step)


def test_agent_applies_its_own_design_choices(tmp_path):
    llm = scripted([
        [("set_fonts", {"body_font": "Times", "heading_font": "Courier"}),
         ("set_colors", {"heading": "#7B1E3A", "accent": "#7B1E3A"}),
         ("set_header_footer", {"header_text": "Agent header", "footer_text": "Agent footer"})],
        [("add_title_block", {"title": "Agent designed report"}),
         ("add_callout", {"text": "Average is 92%", "kind": "success"}),
         ("add_markdown", {"markdown": TURN.answer}),
         ("add_bar_chart", {"labels": ["A", "B", "C"], "values": [92, 95, 89]})],
        [("review_report", {})],
    ])
    result = ReportAgent(llm, Settings()).generate([TURN], tmp_path / "r.pdf", instructions="serif body, wine headings")
    text = pdf_text(result.path)
    assert not result.used_fallback and result.pages == 1
    assert "Agent designed report" in text and "Agent header" in text and "Average is 92%" in text and "Paper C is lowest" in text
    assert {"Times-Roman", "Courier-Bold"} <= pdf_fonts(result.path)
    assert any(line.startswith("set_fonts") for line in result.design_log)


def test_agent_recovers_from_a_tool_error(tmp_path):
    llm = scripted([
        [("set_fonts", {"body_font": "Comic Sans"}), ("set_colors", {"accent": "nonsense"})],  # both invalid
        [("add_title_block", {"title": "Recovered"}), ("add_paragraph", {"text": "still works"})],
    ])
    result = ReportAgent(llm, Settings()).generate([TURN], tmp_path / "r.pdf")
    assert sum("Tool error" in line for line in result.design_log) == 2
    assert "Recovered" in pdf_text(result.path) and not result.used_fallback


def test_agent_falls_back_when_the_model_adds_no_content(tmp_path):
    result = ReportAgent(scripted([]), Settings()).generate([TURN], tmp_path / "r.pdf", title="Fallback report")
    text = pdf_text(result.path)
    assert result.used_fallback
    for expected in ["Fallback report", "What is the average CNN accuracy?", "mean(92, 95, 89) = 92.0", "paper_a_cnn.txt"]:
        assert expected in text


def test_agent_falls_back_when_the_model_raises(tmp_path):
    def boom(messages):
        raise RuntimeError("API down")
    result = ReportAgent(ScriptedLLM(tool_step=boom), Settings()).generate([TURN], tmp_path / "r.pdf")
    assert result.used_fallback and any("API down" in line for line in result.design_log) and result.path.exists()


def test_agent_ignores_unknown_tools(tmp_path):
    llm = scripted([[("delete_everything", {})], [("add_paragraph", {"text": "ok"})]])
    result = ReportAgent(llm, Settings()).generate([TURN], tmp_path / "r.pdf")
    assert any("Unknown tool" in line for line in result.design_log) and "ok" in pdf_text(result.path)


def test_agent_needs_material(tmp_path):
    with pytest.raises(ValueError):
        ReportAgent(scripted([]), Settings()).generate([], tmp_path / "r.pdf")


def test_material_contains_everything_the_agent_needs():
    block = format_material([TURN])
    for part in ["QUESTION:", "ANSWER", "ANALYSIS:", "CALCULATIONS:", "COMPARISONS:", "SOURCES:", "paper_a_cnn.txt"]:
        assert part in block


def test_default_layout_handles_missing_information(tmp_path):
    turn = ReportTurn(question="q?", answer="partial", missing_information=["computational cost"])
    d = ReportDesigner("T")
    build_default_report(d, [turn], "T")
    d.render(tmp_path / "r.pdf")
    assert "computational cost" in pdf_text(tmp_path / "r.pdf")
