"""Shared data models passed between the pipeline, the three agents and the orchestrator."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """One retrieved chunk of a user document, with the metadata needed for citations."""

    chunk_id: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_type: str = "text"  # "text" | "table"
    content: str
    score: float = 0.0


class RewrittenQuery(BaseModel):
    """Output of the Query Rewriter tool."""

    standalone_query: str = Field(description="The question rewritten so it makes sense without chat history.")
    alternative_queries: list[str] = Field(
        default_factory=list, description="Up to 2 differently worded queries with synonyms / expanded terms."
    )


class MetadataFilter(BaseModel):
    """Output of the Metadata Filter tool. Only set a field when the user explicitly restricts scope."""

    source: Optional[str] = Field(default=None, description="Exact document file name the user restricts to.")
    page_min: Optional[int] = Field(default=None, description="First page of a requested page range.")
    page_max: Optional[int] = Field(default=None, description="Last page of a requested page range.")
    chunk_type: Optional[Literal["text", "table"]] = Field(
        default=None, description="'table' only if the user explicitly asks about a table."
    )

    def is_empty(self) -> bool:
        return not any([self.source, self.page_min, self.page_max, self.chunk_type])

    def describe(self) -> str:
        parts = []
        if self.source:
            parts.append(f"source={self.source}")
        if self.page_min is not None or self.page_max is not None:
            parts.append(f"pages={self.page_min or '*'}-{self.page_max or '*'}")
        if self.chunk_type:
            parts.append(f"type={self.chunk_type}")
        return ", ".join(parts) or "none"


class RetrievalResult(BaseModel):
    """What the Retriever Agent hands over: evidence only, never an answer."""

    standalone_query: str
    queries: list[str]
    metadata_filter: MetadataFilter = Field(default_factory=MetadataFilter)
    evidence: list[Evidence] = Field(default_factory=list)
    candidates_considered: int = 0
    trace: list[str] = Field(default_factory=list)


class IntentDecision(BaseModel):
    """Orchestrator's 'understand intent' step."""

    needs_documents: bool = Field(
        description="True if answering requires looking at the uploaded documents. "
        "False for greetings, thanks, or questions about what the assistant can do."
    )
    direct_reply: Optional[str] = Field(
        default=None, description="Short reply to give when needs_documents is false."
    )


class AnalystReport(BaseModel):
    """What the Analyst Agent hands to the Answer Agent (or back to the Retriever via the feedback loop)."""

    sufficient: bool = Field(
        description="True only if the evidence fully supports an answer to every part of the question."
    )
    analysis: str = Field(description="Reasoning over the evidence, including figures and comparisons.")
    calculations: list[str] = Field(
        default_factory=list, description="Each calculation performed, with its result."
    )
    comparisons: list[str] = Field(default_factory=list, description="Key similarities / differences found.")
    used_evidence: list[str] = Field(
        default_factory=list, description="Evidence ids (like E1, E3) the analysis relies on."
    )
    missing_information: list[str] = Field(
        default_factory=list, description="What is missing when sufficient is false."
    )
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description="1-3 specific search queries for the Retriever when sufficient is false.",
    )


class Citation(BaseModel):
    number: int
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_ids: list[str] = Field(default_factory=list)

    def label(self) -> str:
        if self.page is not None:
            return f"{self.source}, page {self.page}"
        if self.section:
            return f"{self.source}, section \"{self.section}\""
        return self.source


class FinalAnswer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)


class ReportTurn(BaseModel):
    """One question/answer exchange with everything the Report Generator Agent may put in a report."""

    question: str
    answer: str
    analysis: str = ""
    calculations: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
