"""Request / response models of the HTTP API."""
from typing import Optional

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    number: int
    label: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None


class TraceOut(BaseModel):
    standalone_question: str = ""
    retriever_queries: list[str] = Field(default_factory=list)
    analyst_tool_calls: list[str] = Field(default_factory=list)
    evidence_sufficient: Optional[bool] = None
    missing_information: list[str] = Field(default_factory=list)
    feedback_loops: int = 0


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None, description="Reuse to keep the conversation (follow-ups, reports).")
    include_trace: bool = False


class AnswerOut(BaseModel):
    session_id: str
    question: str
    answer: str = Field(description="Markdown answer with [n] citation markers and a Sources list.")
    citations: list[CitationOut] = Field(default_factory=list)
    trace: Optional[TraceOut] = None


class OcrOut(BaseModel):
    text: str
    confidence: float = Field(description="Mean word confidence, 0-100.")
    word_count: int
    language: str
    warnings: list[str] = Field(default_factory=list)


class VoiceAnswerOut(AnswerOut):
    transcript: str


class ImageAnswerOut(AnswerOut):
    ocr: OcrOut


class TranscriptOut(BaseModel):
    text: str


class IngestOut(BaseModel):
    source: str
    ok: bool
    pages: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class ImageIngestOut(IngestOut):
    ocr_text: str = ""
    ocr_confidence: float = 0.0


class DocumentInfo(BaseModel):
    chunks: int = 0
    tables: int = 0


class DocumentsOut(BaseModel):
    documents: dict[str, DocumentInfo]


class ReportIn(BaseModel):
    session_id: str = Field(description="The conversation to report on (returned by /query, /voice/query, /images/query).")
    instructions: str = Field(
        default="", max_length=3000,
        description="Design/content wishes in plain language, e.g. 'navy headings, Times body 11pt, two columns, landscape'.",
    )
    title: Optional[str] = Field(default=None, max_length=200)
    question: Optional[str] = Field(default=None, max_length=4000, description="Optional new question to answer first and include.")
    last_n: Optional[int] = Field(default=None, ge=1, le=50, description="Only the last N exchanges (default: all).")


class ReportOut(BaseModel):
    report_id: str
    title: str
    pages: int
    exchanges: int
    download_url: str
    used_default_layout: bool = Field(description="True if the agent produced no layout and the default design was used.")
    design_log: list[str] = Field(default_factory=list, description="Design/formatting tool calls the agent made.")


class HealthOut(BaseModel):
    status: str
    ready: bool
    detail: Optional[str] = None
    documents: int = 0
    auth_required: bool = False
    ocr_available: bool = False
