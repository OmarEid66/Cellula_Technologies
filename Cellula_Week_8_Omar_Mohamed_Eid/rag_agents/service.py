"""Application service: everything the API (or any other front end) needs, with no web-framework code.

Wraps the RAG system (3 agents) and adds the Task 5 features:
  * voice queries      - speech-to-text, then the normal question pipeline
  * image upload + OCR - text is extracted with Tesseract and ingested (or used as the question)
  * report generation  - the Report Generator Agent writes a designed PDF from a session's answers
"""
from __future__ import annotations

import re
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from .agents.report import ReportAgent, ReportResult
from .config import Settings
from .ingestion import IngestReport
from .ingestion.loader import SUPPORTED_EXTENSIONS, load_file
from .ingestion.ocr import OcrResult, is_image_file, ocr_image
from .ingestion.pipeline import ingest_loaded
from .schemas import ReportTurn
from .system import AskResult, RagSystem
from .voice import OpenAITranscriber, Transcriber, validate_audio

MAX_QUESTION_CHARS = 4000
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_REPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SOURCES_TAIL = re.compile(r"\n*\*\*Sources\*\*\s*\n(?:- .*\n?)*\s*$")


@dataclass
class Turn:
    question: str
    result: AskResult
    via: str = "text"  # text | voice | image

    def to_report_turn(self) -> ReportTurn:
        a = self.result.analysis
        answer = _SOURCES_TAIL.sub("", self.result.text).strip()  # sources are given to the report separately
        return ReportTurn(
            question=self.question,
            answer=answer,
            analysis=a.analysis if a else "",
            calculations=list(a.calculations) if a else [],
            comparisons=list(a.comparisons) if a else [],
            sources=[c.label() for c in self.result.answer.citations],
            missing_information=list(a.missing_information) if a and not a.sufficient else [],
        )


class SessionStore:
    """In-memory conversations (history for follow-up questions + material for reports). Bounded and thread-safe."""

    def __init__(self, max_sessions: int = 200, max_turns: int = 50):
        self._sessions: OrderedDict[str, list[Turn]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions, self.max_turns = max_sessions, max_turns

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def validate_id(session_id: str) -> str:
        if not _SESSION_RE.match(session_id or ""):
            raise ValueError("Invalid session_id (use letters, digits, '-' or '_', max 64 characters).")
        return session_id

    def history(self, session_id: str) -> list[dict]:
        with self._lock:
            turns = list(self._sessions.get(session_id, []))
        out: list[dict] = []
        for t in turns:
            out += [{"role": "user", "content": t.question}, {"role": "assistant", "content": t.result.text}]
        return out

    def turns(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            self._sessions.move_to_end(session_id)
            turns.append(turn)
            del turns[: max(0, len(turns) - self.max_turns)]
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


@dataclass
class ImageIngestResult:
    report: IngestReport
    ocr_text: str
    confidence: float
    word_count: int


@dataclass
class ReportInfo:
    report_id: str
    result: ReportResult
    turns_used: int = 0


class AssistantService:
    def __init__(
        self,
        rag: RagSystem | None = None,
        settings: Settings | None = None,
        transcriber: Transcriber | None = None,
        report_agent: ReportAgent | None = None,
    ):
        self.settings = settings or (rag.settings if rag else Settings())
        self.rag = rag or RagSystem(self.settings)
        self.transcriber = transcriber or OpenAITranscriber(self.settings)
        self.report_agent = report_agent or ReportAgent(self.rag.llm, self.settings)
        self.sessions = SessionStore()
        self._ingest_lock = threading.Lock()
        self.reports_dir = Path(self.settings.reports_dir)
        self.max_upload_bytes = self.settings.max_upload_mb * 1024 * 1024

    # ------------------------------------------------------------ documents
    @staticmethod
    def safe_name(filename: str) -> str:
        name = Path((filename or "").replace("\\", "/")).name.strip().strip(".")
        name = re.sub(r"[^\w.\- ()]+", "_", name, flags=re.UNICODE)
        if not name:
            raise ValueError("The uploaded file has no usable name.")
        return name[:150]

    def ingest_bytes(self, files: list[tuple[str, bytes]]) -> list[IngestReport]:
        """Ingest uploaded files (documents and images). One bad file never stops the others."""
        reports: list[IngestReport] = []
        with tempfile.TemporaryDirectory() as tmp:
            for filename, data in files:
                try:
                    name = self.safe_name(filename)
                except ValueError as exc:
                    reports.append(IngestReport(source=filename or "(unnamed)", error=str(exc)))
                    continue
                ext = Path(name).suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    reports.append(IngestReport(source=name, error=f"unsupported file type '{ext or 'none'}'"))
                    continue
                if not data:
                    reports.append(IngestReport(source=name, error="the file is empty"))
                    continue
                path = Path(tmp) / name
                path.write_bytes(data)
                try:
                    loaded = load_file(path, source_name=name, ocr_languages=self.settings.ocr_languages,
                                       ocr_min_confidence=self.settings.ocr_min_confidence)
                except Exception as exc:
                    reports.append(IngestReport(source=name, error=f"{type(exc).__name__}: {exc}"))
                    continue
                with self._ingest_lock:
                    reports.append(ingest_loaded(loaded, self.rag.kb, self.settings))
        return reports

    def documents(self) -> dict[str, dict[str, int]]:
        return self.rag.documents()

    def remove_document(self, source: str) -> bool:
        if source not in self.rag.documents():
            return False
        with self._ingest_lock:
            self.rag.remove_document(source)
        return True

    def clear_documents(self) -> None:
        with self._ingest_lock:
            self.rag.clear()
        self.sessions.clear()

    # ------------------------------------------------------------ questions
    def _session(self, session_id: str | None) -> str:
        return self.sessions.validate_id(session_id) if session_id else self.sessions.new_id()

    @staticmethod
    def _clean_question(question: str) -> str:
        q = (question or "").strip()
        if not q:
            raise ValueError("The question is empty.")
        if len(q) > MAX_QUESTION_CHARS:
            raise ValueError(f"The question is too long (limit {MAX_QUESTION_CHARS} characters).")
        return q

    def ask(self, question: str, session_id: str | None = None, via: str = "text") -> tuple[str, AskResult]:
        q = self._clean_question(question)
        sid = self._session(session_id)
        result = self.rag.ask(q, history=self.sessions.history(sid))
        self.sessions.add(sid, Turn(q, result, via))
        return sid, result

    # ------------------------------------------------------------ voice
    def transcribe(self, audio: bytes, filename: str, language: str | None = None) -> str:
        name = validate_audio(audio, filename, self.max_upload_bytes)
        return self.transcriber.transcribe(audio, name, language)

    def ask_voice(self, audio: bytes, filename: str, session_id: str | None = None,
                  language: str | None = None) -> tuple[str, str, AskResult]:
        """speech -> text -> question pipeline -> answer. Returns (transcript, session_id, result)."""
        transcript = self.transcribe(audio, filename, language)
        sid, result = self.ask(transcript, session_id, via="voice")
        return transcript, sid, result

    # ------------------------------------------------------------ images / OCR
    def ocr(self, image: bytes, filename: str = "", languages: str | None = None) -> OcrResult:
        self._check_image(image, filename)
        return ocr_image(image, languages or self.settings.ocr_languages, self.settings.ocr_min_confidence)

    def _check_image(self, image: bytes, filename: str) -> None:
        if not image:
            raise ValueError("The image is empty.")
        if len(image) > self.max_upload_bytes:
            raise ValueError(f"The image is too large (limit {self.settings.max_upload_mb} MB).")
        if filename and not is_image_file(filename):
            raise ValueError(f"'{filename}' is not a supported image type.")

    def ingest_image(self, image: bytes, filename: str) -> ImageIngestResult:
        """OCR the image once and add its text to the knowledge base like any other document."""
        self._check_image(image, filename)
        name = self.safe_name(filename or "image.png")
        if not is_image_file(name):
            raise ValueError(f"'{name}' is not a supported image type.")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(image)
            loaded = load_file(path, source_name=name, ocr_languages=self.settings.ocr_languages,
                               ocr_min_confidence=self.settings.ocr_min_confidence)
        text = loaded.pages[0].text if loaded.pages else ""
        with self._ingest_lock:
            report = ingest_loaded(loaded, self.rag.kb, self.settings)
        return ImageIngestResult(report=report, ocr_text=text, confidence=loaded.ocr_confidence or 0.0,
                                 word_count=len(text.split()))

    def ask_from_image(self, image: bytes, filename: str, extra_question: str = "",
                       session_id: str | None = None) -> tuple[OcrResult, str, AskResult]:
        """The text in the image becomes the question (optionally combined with a typed instruction)."""
        result = self.ocr(image, filename)
        if not result.has_text:
            raise ValueError("No text could be read from the image, so there is no question to ask.")
        text = result.text[:MAX_QUESTION_CHARS - 200]
        question = f"{extra_question.strip()}\n\nText from the image:\n{text}" if extra_question.strip() else text
        sid, answer = self.ask(question, session_id, via="image")
        return result, sid, answer

    # ------------------------------------------------------------ reports
    def create_report(self, session_id: str, instructions: str = "", title: str | None = None,
                      question: str | None = None, last_n: int | None = None) -> ReportInfo:
        sid = self.sessions.validate_id(session_id)
        if question and question.strip():  # run a fresh question first and report on it as well
            self.ask(question, sid)
        turns = self.sessions.turns(sid)
        if last_n:
            turns = turns[-last_n:]
        if not turns:
            raise ValueError("Nothing to report yet: ask at least one question in this session first.")
        report_id = uuid.uuid4().hex
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        result = self.report_agent.generate(
            [t.to_report_turn() for t in turns],
            self.reports_dir / f"{report_id}.pdf",
            instructions=(instructions or "")[:3000],
            title=title,
        )
        return ReportInfo(report_id=report_id, result=result, turns_used=len(turns))

    def report_path(self, report_id: str) -> Path | None:
        if not _REPORT_ID_RE.match(report_id or ""):
            return None
        path = self.reports_dir / f"{report_id}.pdf"
        return path if path.is_file() else None
