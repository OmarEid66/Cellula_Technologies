"""High-level facade: ingest documents and chat with them through the 3-agent LangGraph workflow."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .agents import AnalystAgent, AnswerAgent, RetrieverAgent
from .config import Settings
from .graph import build_graph
from .ingestion import IngestReport, ingest_files
from .llm import get_embeddings, get_llm
from .schemas import AnalystReport, Evidence, FinalAnswer, RetrievalResult
from .vectorstore import KnowledgeBase


@dataclass
class AskResult:
    answer: FinalAnswer
    standalone_question: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    retrievals: list[RetrievalResult] = field(default_factory=list)
    analysis: AnalystReport | None = None
    analyst_trace: list[list[str]] = field(default_factory=list)
    feedback_loops: int = 0

    @property
    def text(self) -> str:
        return self.answer.text


class RagSystem:
    def __init__(self, settings: Settings | None = None, llm=None, embeddings=None):
        self.settings = settings or Settings()
        self.llm = llm or get_llm(self.settings)
        self.embeddings = embeddings or get_embeddings(self.settings)
        self.kb = KnowledgeBase(self.embeddings, self.settings.chroma_dir, self.settings.collection_name)
        self.retriever = RetrieverAgent(self.llm, self.kb, self.settings)
        self.analyst = AnalystAgent(self.llm, self.settings)
        self.answerer = AnswerAgent(self.llm)
        self.graph = build_graph(
            self.retriever, self.analyst, self.answerer, self.llm, self.settings, lambda: self.kb.count() > 0
        )

    # ---- documents ----
    def ingest(self, paths: list[str | Path]) -> list[IngestReport]:
        return ingest_files(paths, self.kb, self.settings)

    def documents(self) -> dict[str, dict[str, int]]:
        return self.kb.list_sources()

    def remove_document(self, source: str) -> None:
        self.kb.delete_source(source)

    def clear(self) -> None:
        self.kb.reset()

    # ---- chat ----
    def _initial_state(self, question: str, history: list[dict] | None) -> dict[str, Any]:
        return {"question": question, "history": history or []}

    def ask(self, question: str, history: list[dict] | None = None) -> AskResult:
        state = self.graph.invoke(self._initial_state(question, history))
        return self._to_result(state)

    def stream(self, question: str, history: list[dict] | None = None) -> Iterator[tuple[str, dict]]:
        """Yields (node_name, state_update) as each step finishes; the last 'answer' update holds the final answer."""
        yield from (
            (node, update)
            for chunk in self.graph.stream(self._initial_state(question, history), stream_mode="updates")
            for node, update in chunk.items()
        )

    @staticmethod
    def _to_result(state: dict[str, Any]) -> AskResult:
        return AskResult(
            answer=state["final"],
            standalone_question=state.get("standalone_question", ""),
            evidence=state.get("evidence", []),
            retrievals=state.get("retrievals", []),
            analysis=state.get("analysis"),
            analyst_trace=state.get("analyst_trace", []),
            feedback_loops=state.get("feedback_loops", 0),
        )
