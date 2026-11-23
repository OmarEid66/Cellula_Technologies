"""End-to-end tests of the LangGraph workflow with a scripted LLM (no network)."""
from langchain_core.messages import AIMessage

from fakes import HashEmbeddings, ScriptedLLM, last_human
from rag_agents.graph import NO_DOCS_REPLY
from rag_agents.system import RagSystem

from conftest import SAMPLES


FOLLOW_UP = "wide residual networks dropout 5.2 GFLOPs"


def make_llm(*, sufficient_after_followup: bool = True, always_insufficient: bool = False):
    state = {"analyses": 0, "tool_round": {}}

    def rewrite(messages):
        q = last_human(messages).split("Question:")[-1].strip()
        return {"standalone_query": q, "alternative_queries": []}

    def rerank(messages):
        import re

        n = len(re.findall(r"^\[\d+\]", last_human(messages), flags=re.M))
        return {"scores": [{"index": i, "score": 10 - i} for i in range(n)]}

    def intent(messages):
        msg = last_human(messages).lower()
        if "thanks" in msg:
            return {"needs_documents": False, "direct_reply": "You're welcome!"}
        return {"needs_documents": True}

    def tool_step(messages):
        # first call: use the calculator; second call: stop
        if not any(getattr(m, "type", "") == "tool" for m in messages):
            return AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "mean(92, 95, 89)"}, "id": "c1"}])
        return AIMessage(content="finished")

    def report(messages):
        state["analyses"] += 1
        blob = " ".join(str(m.content) for m in messages)
        if always_insufficient or state["analyses"] == 1:
            return {
                "sufficient": False,
                "analysis": "Accuracy values known, computational cost missing.",
                "used_evidence": ["E1"],
                "missing_information": ["computational cost"],
                "follow_up_queries": [FOLLOW_UP],
            }
        return {
            "sufficient": True,
            "analysis": "Accuracies are 92, 95, 89 [E1]; mean is 92.",
            "calculations": ["mean(92, 95, 89) = 92.0"],
            "used_evidence": ["E1", "E2", "E99"],
        }

    def answer(messages):
        return "The average CNN accuracy is 92% [1][2][9]."

    llm = ScriptedLLM(
        structured={"RewrittenQuery": rewrite, "_RerankResult": rerank, "IntentDecision": intent, "AnalystReport": report},
        tool_step=tool_step,
        text=answer,
    )
    return llm, state


def system(tmp_settings, llm):
    rag = RagSystem(settings=tmp_settings, llm=llm, embeddings=HashEmbeddings())
    rag.ingest(sorted(SAMPLES.glob("*.txt")))
    return rag


def test_no_documents_short_circuits(settings):
    llm, _ = make_llm()
    rag = RagSystem(settings=settings, llm=llm, embeddings=HashEmbeddings())
    result = rag.ask("anything?")
    assert result.text == NO_DOCS_REPLY and llm.calls == []


def test_small_talk_skips_agents(settings):
    llm, _ = make_llm()
    rag = system(settings, llm)
    result = rag.ask("thanks!")
    assert result.text == "You're welcome!" and result.retrievals == []


def test_full_flow_with_feedback_loop(settings):
    settings.context_chunks = 1  # one chunk per retrieval, so the follow-up search has something new to find
    llm, state = make_llm()
    rag = system(settings, llm)
    result = rag.ask("shallow baseline convolutional network Adam optimizer")
    assert state["analyses"] == 2 and result.feedback_loops == 1  # insufficient -> retrieve more -> sufficient
    assert len(result.retrievals) == 2  # initial + one follow-up
    assert result.retrievals[1].standalone_query == FOLLOW_UP
    assert any("calculator" in line and "= 92.0" in line for line in result.analyst_trace[0])
    assert result.analysis.sufficient
    # evidence ids invalid (E99) are dropped, invalid citation [9] removed, sources appended
    assert "[9]" not in result.text and "[1][2]" in result.text
    assert "**Sources**" in result.text and ".txt" in result.text
    assert len({e.chunk_id for e in result.evidence}) == len(result.evidence)  # no duplicate evidence


def test_loop_is_bounded(settings):
    settings.max_feedback_loops = 2
    settings.context_chunks = 1
    llm, state = make_llm(always_insufficient=True)
    rag = system(settings, llm)
    result = rag.ask("shallow baseline convolutional network Adam optimizer")
    assert result.feedback_loops <= 2 and state["analyses"] <= 3
    assert result.text  # still produces an answer


def test_stream_emits_node_updates(settings):
    llm, _ = make_llm()
    rag = system(settings, llm)
    nodes = [n for n, _ in rag.stream("What is the average CNN accuracy?")]
    assert nodes[0] == "understand_intent" and nodes[-1] == "answer" and "analyze" in nodes
    state = {}
    for _, upd in rag.stream("What is the average CNN accuracy?"):
        state.update(upd)
    assert RagSystem._to_result(state).text


def test_documents_management(settings):
    llm, _ = make_llm()
    rag = system(settings, llm)
    assert set(rag.documents()) == {"paper_a_cnn.txt", "paper_b_cnn.txt", "paper_c_cnn.txt"}
    rag.remove_document("paper_a_cnn.txt")
    assert "paper_a_cnn.txt" not in rag.documents()
    rag.clear()
    assert rag.documents() == {}


def test_stalled_follow_up_goes_straight_to_answer(settings):
    """If the follow-up search finds nothing new, the loop stops instead of re-analysing identical evidence."""
    llm, state = make_llm()  # default context_chunks=6 -> the first retrieval already holds every chunk
    rag = system(settings, llm)
    result = rag.ask("What is the average CNN accuracy across the three papers?")
    assert state["analyses"] == 1 and result.text
