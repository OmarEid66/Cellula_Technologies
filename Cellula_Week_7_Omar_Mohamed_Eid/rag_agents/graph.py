"""Orchestrator: a LangGraph workflow wiring User -> Orchestrator -> Retriever -> Analyst -> (feedback loop) -> Answer."""
from __future__ import annotations

from typing import Callable, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .agents import AnalystAgent, AnswerAgent, RetrieverAgent
from .config import Settings
from .formatting import format_history
from .llm import structured
from .schemas import AnalystReport, Evidence, FinalAnswer, IntentDecision, RetrievalResult


class GraphState(TypedDict, total=False):
    question: str
    history: list[dict]
    needs_documents: bool
    standalone_question: str
    evidence: list[Evidence]
    retrievals: list[RetrievalResult]
    analysis: Optional[AnalystReport]
    analyst_trace: list[list[str]]
    follow_up_queries: list[str]
    feedback_loops: int
    stalled: bool
    final: FinalAnswer


_INTENT_SYSTEM = (
    "You are the Orchestrator of a document question-answering assistant. The user has uploaded documents. "
    "Decide whether the latest message needs the documents to be answered (needs_documents=true) or is "
    "small talk / a thanks / a question about what you can do (needs_documents=false, with a short, friendly "
    "direct_reply). When unsure, choose needs_documents=true."
)

NO_DOCS_REPLY = "No documents have been uploaded yet. Please upload a PDF, Word or text file first, then ask your question."


def build_graph(
    retriever: RetrieverAgent,
    analyst: AnalystAgent,
    answerer: AnswerAgent,
    llm,
    settings: Settings,
    has_documents: Callable[[], bool],
):
    # ---------------- nodes ----------------
    def understand_intent(state: GraphState) -> GraphState:
        if not has_documents():
            return {"needs_documents": False, "final": FinalAnswer(text=NO_DOCS_REPLY)}
        try:
            decision = structured(llm, IntentDecision).invoke(
                [
                    SystemMessage(content=_INTENT_SYSTEM),
                    HumanMessage(
                        content=f"Chat history:\n{format_history(state.get('history'))}\n\nLatest message: {state['question']}"
                    ),
                ]
            )
            if isinstance(decision, dict):
                decision = IntentDecision(**decision)
        except Exception:
            decision = IntentDecision(needs_documents=True)
        if decision.needs_documents:
            return {"needs_documents": True, "feedback_loops": 0, "evidence": [], "retrievals": [], "analyst_trace": []}
        reply = decision.direct_reply or "Hello! Ask me anything about your uploaded documents."
        return {"needs_documents": False, "final": FinalAnswer(text=reply)}

    def retrieve(state: GraphState) -> GraphState:
        evidence = list(state.get("evidence", []))
        known = {e.chunk_id for e in evidence}
        retrievals = list(state.get("retrievals", []))
        loops = state.get("feedback_loops", 0)
        update: GraphState = {}

        if state.get("analysis") is None:  # first pass: retrieve for the user's question
            result = retriever.retrieve(state["question"], state.get("history"))
            retrievals.append(result)
            update["standalone_question"] = result.standalone_query
            new = [e for e in result.evidence if e.chunk_id not in known]
            evidence.extend(new)
        else:  # feedback loop: retrieve for the Analyst's follow-up queries
            loops += 1
            added = 0
            for q in state.get("follow_up_queries", [])[:3]:
                result = retriever.retrieve(q, None)
                retrievals.append(result)
                for e in result.evidence:
                    if e.chunk_id not in known:
                        known.add(e.chunk_id)
                        evidence.append(e)
                        added += 1
            update["stalled"] = added == 0
            update["feedback_loops"] = loops
        update["evidence"] = evidence
        update["retrievals"] = retrievals
        return update

    def analyze(state: GraphState) -> GraphState:
        question = state.get("standalone_question") or state["question"]
        report, trace = analyst.analyze(question, state.get("evidence", []))
        return {
            "analysis": report,
            "follow_up_queries": report.follow_up_queries,
            "analyst_trace": [*state.get("analyst_trace", []), trace],
            "stalled": False,
        }

    def answer(state: GraphState) -> GraphState:
        final = answerer.answer(
            state.get("standalone_question") or state["question"],
            state.get("evidence", []),
            state.get("analysis"),
            state.get("history"),
        )
        return {"final": final}

    # ---------------- routing ----------------
    def after_intent(state: GraphState) -> str:
        return "retrieve" if state.get("needs_documents") else END

    def after_retrieve(state: GraphState) -> str:
        # a follow-up search that found nothing new cannot change the analysis -> stop looping
        return "answer" if state.get("stalled") else "analyze"

    def evidence_assessment(state: GraphState) -> str:
        report = state.get("analysis")
        if report is None or report.sufficient:
            return "answer"
        if state.get("follow_up_queries") and state.get("feedback_loops", 0) < settings.max_feedback_loops:
            return "retrieve"  # FEEDBACK LOOP: not enough evidence -> ask the Retriever for more
        return "answer"

    g = StateGraph(GraphState)
    g.add_node("understand_intent", understand_intent)
    g.add_node("retrieve", retrieve)
    g.add_node("analyze", analyze)
    g.add_node("answer", answer)
    g.add_edge(START, "understand_intent")
    g.add_conditional_edges("understand_intent", after_intent, {"retrieve": "retrieve", END: END})
    g.add_conditional_edges("retrieve", after_retrieve, {"analyze": "analyze", "answer": "answer"})
    g.add_conditional_edges("analyze", evidence_assessment, {"retrieve": "retrieve", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()
