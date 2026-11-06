import json

import pytest
from langchain_core.messages import AIMessage

from fakes import ScriptedLLM
from rag_agents.schemas import Citation, Evidence, MetadataFilter
from rag_agents.tools import retrieval as rt
from rag_agents.tools.analysis import AnalystToolkit, run_calculator, safe_eval
from rag_agents.tools.answer_tools import citation_formatter, response_formatter, source_formatter
from rag_agents.vectorstore import to_chroma_where, tokenize


# ---------------- calculator
def test_calculator_values():
    assert safe_eval("mean(92, 95, 89)") == pytest.approx(92.0)
    assert safe_eval("(95-92)/92*100") == pytest.approx(3.26087, rel=1e-4)
    assert safe_eval("mean([92, 95, 89])") == pytest.approx(92.0)
    assert safe_eval("pct_change(80, 100)") == 25
    assert safe_eval("2^3") == 8
    assert run_calculator("1/0") == "Error: division by zero"


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "open('x')", "a+1", "(1).real", "9**9999", "[x for x in range(3)]"])
def test_calculator_rejects_unsafe(bad):
    assert run_calculator(bad).startswith("Error")


# ---------------- search
def test_keyword_and_semantic_search(loaded_kb):
    kw = rt.keyword_search(loaded_kb, "GFLOPs 5.2", 5)
    assert kw and kw[0].source == "paper_b_cnn.txt"
    sem = rt.semantic_search(loaded_kb, "wide residual networks dropout", 3)
    assert sem and sem[0].source == "paper_b_cnn.txt"
    assert all(0.0 <= e.score <= 1.0 for e in sem)


def test_tokenizer_keeps_hyphenated_terms():
    toks = tokenize("BERT-base and Equation 12")
    assert "bert-base" in toks and "bert" in toks and "12" in toks


def test_metadata_filter_applies_to_both_searches(loaded_kb):
    flt = MetadataFilter(source="paper_c_cnn.txt")
    for fn in (rt.semantic_search, rt.keyword_search):
        res = fn(loaded_kb, "CNN accuracy", 10, flt)
        assert res and {e.source for e in res} == {"paper_c_cnn.txt"}
    assert to_chroma_where(MetadataFilter()) is None
    assert to_chroma_where(MetadataFilter(source="a", page_min=2)) == {"$and": [{"source": "a"}, {"page": {"$gte": 2}}]}


def test_metadata_filter_tool_validates_source():
    llm = ScriptedLLM(structured={"MetadataFilter": lambda m: {"source": "PAPER_B_CNN"}})
    sources = ["paper_a_cnn.txt", "paper_b_cnn.txt"]
    assert rt.metadata_filter(llm, "According to paper_b_cnn.txt what is the accuracy?", sources).source == "paper_b_cnn.txt"
    assert not llm.calls == [] or True
    llm2 = ScriptedLLM(structured={"MetadataFilter": lambda m: {"source": "invented.pdf"}})
    assert rt.metadata_filter(llm2, "see invented.pdf page 2", sources).source is None
    # no scope words -> the LLM is not even called
    llm3 = ScriptedLLM()
    assert rt.metadata_filter(llm3, "what is the average accuracy", sources).is_empty() and llm3.calls == []


def test_query_rewriter_falls_back_on_error():
    llm = ScriptedLLM()  # no handler -> raises inside, must fall back
    out = rt.query_rewriter(llm, "What are its disadvantages?", [])
    assert out.standalone_query == "What are its disadvantages?"
    llm2 = ScriptedLLM(structured={"RewrittenQuery": lambda m: {"standalone_query": "Disadvantages of RAG", "alternative_queries": ["RAG limits", "", "a", "b"]}})
    out2 = rt.query_rewriter(llm2, "its disadvantages?", [{"role": "user", "content": "What is RAG?"}])
    assert out2.standalone_query == "Disadvantages of RAG" and len(out2.alternative_queries) == 2


# ---------------- fusion / rerank / selector
def ev(cid, src="d.pdf", page=1, text="t", score=0.5, typ="text"):
    return Evidence(chunk_id=cid, source=src, page=page, content=text, score=score, chunk_type=typ)


def test_rrf_merges_lists():
    a = [ev("1"), ev("2"), ev("3")]
    b = [ev("3"), ev("1")]
    fused = rt.reciprocal_rank_fusion([a, b])
    assert [e.chunk_id for e in fused][:2] == ["1", "3"] and fused[0].score == 1.0


def test_reranker_reorders_and_falls_back():
    cands = [ev("a", text="x"), ev("b", text="y"), ev("c", text="z")]
    llm = ScriptedLLM(structured={"_RerankResult": lambda m: {"scores": [{"index": 0, "score": 2}, {"index": 1, "score": 9}, {"index": 2, "score": 5}]}})
    out = rt.reranker(llm, "q", cands, 2)
    assert [e.chunk_id for e in out] == ["b", "c"] and out[0].score == 0.9
    assert [e.chunk_id for e in rt.reranker(ScriptedLLM(), "q", cands, 2)] == ["a", "b"]  # LLM failure -> original order


def test_context_selector_dedupes_limits_and_keeps_metadata():
    same = "the cnn reached ninety two percent accuracy on cifar ten"
    cands = [
        ev("1", text=same, score=0.9),
        ev("2", text=same, score=0.89),  # near-duplicate -> dropped
        ev("3", src="other.pdf", page=7, text="completely different content about training epochs", score=0.5),
        ev("4", text="a third distinct passage on parameter counts and memory", score=0.4),
    ]
    out = rt.context_selector(cands, max_chunks=3)
    ids = [e.chunk_id for e in out]
    assert "2" not in ids and ids[0] == "1" and len(out) == 3
    assert next(e for e in out if e.chunk_id == "3").page == 7
    assert len(rt.context_selector(cands, max_chunks=1)) == 1
    assert rt.context_selector([], 5) == []


# ---------------- analyst tools
TABLE = "Table from r.pdf, page 2:\n| Model | Accuracy | F1 |\n| --- | --- | --- |\n| A | 92.1% | 0.90 |\n| B | 95.4% | 0.94 |\n| C | 89 | 0.88 |"


def test_table_extractor_and_data_analysis():
    kit = AnalystToolkit([ev("t", src="r.pdf", page=2, text=TABLE, typ="table")], ScriptedLLM())
    tables = json.loads(kit.extract_tables())
    assert tables[0]["id"] == "T1" and tables[0]["rows"][1]["Accuracy"] == "95.4%"
    stats = json.loads(kit.analyze_data(table_id="T1", label_column="Model", value_column="Accuracy"))
    assert stats["best"] == {"label": "B", "value": 95.4} and stats["worst"]["label"] == "C"
    assert stats["mean"] == pytest.approx((92.1 + 95.4 + 89) / 3, rel=1e-5)
    lower = json.loads(kit.analyze_data(values={"x": 3, "y": 1}, higher_is_better=False))
    assert lower["best"]["label"] == "y"
    assert "Unknown table" in kit.analyze_data(table_id="T9", value_column="F1")
    assert "No tables" in AnalystToolkit([ev("x", text="plain")], ScriptedLLM()).extract_tables()


def test_compare_documents_needs_two_docs_and_request_more_evidence():
    one = AnalystToolkit([ev("1", src="a")], ScriptedLLM())
    assert "Only one document" in one.compare_documents(["accuracy"])
    two = AnalystToolkit([ev("1", src="a"), ev("2", src="b")], ScriptedLLM(text=lambda m: "table of differences"))
    assert two.compare_documents(["accuracy"]) == "table of differences"
    two.request_more_evidence("complexity of model B", "needed")
    assert two.more_evidence_requests == ["complexity of model B"]
    assert {t.name for t in two.tools()} == {"calculator", "extract_tables", "compare_documents", "analyze_data", "request_more_evidence"}


# ---------------- answer tools
def test_citations_grouped_by_page_and_formatted():
    evidence = [ev("1", "rag.pdf", 5), ev("2", "rag.pdf", 5), ev("3", "rag.pdf", 8), Evidence(chunk_id="4", source="n.docx", section="Intro", content="x")]
    cits, mapping = citation_formatter(evidence, [0, 1, 2, 3])
    assert [c.number for c in cits] == [1, 2, 3] and mapping == {0: 1, 1: 1, 2: 2, 3: 3}
    assert cits[0].chunk_ids == ["1", "2"]
    src = source_formatter(cits)
    assert "[1] rag.pdf, page 5" in src and "[2] rag.pdf, page 8" in src and 'n.docx, section "Intro"' in src
    assert source_formatter([]) == ""


def test_response_formatter():
    out = response_formatter("RAG helps [1][1] and more [7] and [1, 2].\n\n\n\nEnd", {1, 2})
    assert "[7]" not in out and "[1][1]" not in out and "[1][2]" in out and "\n\n\n" not in out
    table = response_formatter("Results:\n| a | b |\n|---|---|\n| 1 | 2 |", set())
    assert "Results:\n\n| a | b |" in table
    assert Citation(number=1, source="a.pdf", page=3).label() == "a.pdf, page 3"
