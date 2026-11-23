"""FastAPI application.

Run:   uvicorn api.main:app --host 127.0.0.1 --port 8000
Docs:  http://127.0.0.1:8000/docs      UI: http://127.0.0.1:8000/
"""
import logging
import secrets
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from rag_agents.config import Settings
from rag_agents.ingestion.ocr import OcrError, OcrResult, OcrUnavailableError
from rag_agents.service import AssistantService
from rag_agents.system import AskResult
from rag_agents.voice import AudioError, SpeechUnavailableError

from .schemas import (AnswerOut, CitationOut, DocumentInfo, DocumentsOut, HealthOut, ImageAnswerOut, ImageIngestOut,
                      IngestOut, OcrOut, QueryIn, ReportIn, ReportOut, TraceOut, TranscriptOut, VoiceAnswerOut)

log = logging.getLogger("api")
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------- helpers


def _service(request: Request) -> AssistantService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(503, detail=f"The assistant is not ready: {request.app.state.startup_error}")
    return service


def require_api_key(request: Request) -> None:
    """If API_KEY is configured, every endpoint (except /health and the UI page) needs the X-API-Key header."""
    expected = request.app.state.settings.api_key
    if expected and not secrets.compare_digest(request.headers.get("x-api-key", ""), expected):
        raise HTTPException(401, detail="Missing or invalid X-API-Key header.")


async def read_upload(file: UploadFile, limit_bytes: int) -> bytes:
    data = await file.read(limit_bytes + 1)  # never buffer more than limit+1 bytes
    if len(data) > limit_bytes:
        raise HTTPException(413, detail=f"'{file.filename}' is larger than the {limit_bytes // (1024 * 1024)} MB limit.")
    return data


def _answer_fields(question: str, sid: str, result: AskResult, include_trace: bool) -> dict:
    trace = None
    if include_trace:
        a = result.analysis
        trace = TraceOut(
            standalone_question=result.standalone_question,
            retriever_queries=[q for r in result.retrievals for q in r.queries],
            analyst_tool_calls=[line for t in result.analyst_trace for line in t],
            evidence_sufficient=a.sufficient if a else None,
            missing_information=list(a.missing_information) if a else [],
            feedback_loops=result.feedback_loops,
        )
    return dict(
        session_id=sid,
        question=question,
        answer=result.text,
        citations=[CitationOut(number=c.number, label=c.label(), source=c.source, page=c.page, section=c.section)
                   for c in result.answer.citations],
        trace=trace,
    )


def _ocr_out(r: OcrResult) -> OcrOut:
    return OcrOut(text=r.text, confidence=r.confidence, word_count=r.word_count, language=r.language, warnings=r.warnings)


def _ingest_out(r) -> IngestOut:
    return IngestOut(source=r.source, ok=r.ok, pages=r.pages, text_chunks=r.text_chunks, table_chunks=r.table_chunks,
                     warnings=r.warnings, error=r.error)


# ---------------------------------------------------------------- application factory


def create_app(service: Optional[AssistantService] = None, settings: Optional[Settings] = None) -> FastAPI:
    """`service` can be injected (tests); otherwise it is built at startup from the environment."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service, app.state.startup_error = service, ""
            app.state.settings = service.settings
        else:
            app.state.settings = settings or Settings()
            try:
                app.state.service = AssistantService(settings=app.state.settings)
                app.state.startup_error = ""
            except Exception as exc:  # e.g. missing OPENAI_API_KEY: keep the server up and explain on every call
                log.exception("Could not start the assistant")
                app.state.service = None
                app.state.startup_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(
        title="Chat with your documents - agentic system",
        version="1.0.0",
        description="Retriever, Analyst and Answer agents over your documents, with voice queries, image OCR and a "
                    "Report Generator agent that produces designed PDF reports.",
        lifespan=lifespan,
    )
    # populated by the lifespan; defined here so attribute access never fails (e.g. in /health before startup ends)
    app.state.service, app.state.startup_error = service, ""
    app.state.settings = service.settings if service else (settings or Settings())

    # ---------- errors
    @app.exception_handler(OcrUnavailableError)
    @app.exception_handler(SpeechUnavailableError)
    async def _unavailable(_: Request, exc: Exception):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(OcrError)
    @app.exception_handler(AudioError)
    @app.exception_handler(ValueError)
    async def _bad_input(_: Request, exc: Exception):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {str(exc)[:300]}"})

    # ---------- public routes
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthOut, tags=["system"])
    def health(request: Request):
        svc = request.app.state.service
        return HealthOut(
            status="ok" if svc else "unavailable",
            ready=svc is not None,
            detail=request.app.state.startup_error or None,
            documents=len(svc.documents()) if svc else 0,
            auth_required=bool(request.app.state.settings.api_key),
            ocr_available=shutil.which("tesseract") is not None,
        )

    router = APIRouter(dependencies=[Depends(require_api_key)])

    # ---------- documents
    @router.get("/documents", response_model=DocumentsOut, tags=["documents"])
    def list_documents(svc: AssistantService = Depends(_service)):
        return DocumentsOut(documents={k: DocumentInfo(**v) for k, v in svc.documents().items()})

    @router.post("/documents", response_model=list[IngestOut], tags=["documents"],
                 summary="Upload PDF / DOCX / TXT / MD files or images (images are OCR-ed)")
    async def upload_documents(files: list[UploadFile] = File(...), svc: AssistantService = Depends(_service)):
        payload = [(f.filename or "", await read_upload(f, svc.max_upload_bytes)) for f in files]
        reports = await run_in_threadpool(svc.ingest_bytes, payload)
        return [_ingest_out(r) for r in reports]

    @router.delete("/documents/{name}", tags=["documents"])
    def delete_document(name: str, svc: AssistantService = Depends(_service)):
        if not svc.remove_document(name):
            raise HTTPException(404, detail=f"No document named '{name}'.")
        return {"deleted": name}

    @router.delete("/documents", tags=["documents"], summary="Remove every document and every conversation")
    def clear_documents(svc: AssistantService = Depends(_service)):
        svc.clear_documents()
        return {"cleared": True}

    # ---------- questions
    @router.post("/query", response_model=AnswerOut, tags=["query"])
    async def query(body: QueryIn, svc: AssistantService = Depends(_service)):
        sid, result = await run_in_threadpool(svc.ask, body.question, body.session_id)
        return AnswerOut(**_answer_fields(body.question.strip(), sid, result, body.include_trace))

    # ---------- voice
    @router.post("/voice/transcribe", response_model=TranscriptOut, tags=["voice"], summary="Speech to text only")
    async def voice_transcribe(audio: UploadFile = File(...), language: Optional[str] = Form(None),
                               svc: AssistantService = Depends(_service)):
        data = await read_upload(audio, svc.max_upload_bytes)
        text = await run_in_threadpool(svc.transcribe, data, audio.filename or "recording.webm", language)
        return TranscriptOut(text=text)

    @router.post("/voice/query", response_model=VoiceAnswerOut, tags=["voice"],
                 summary="Ask by voice: audio -> text -> answer")
    async def voice_query(audio: UploadFile = File(...), session_id: Optional[str] = Form(None),
                          language: Optional[str] = Form(None), include_trace: bool = Form(False),
                          svc: AssistantService = Depends(_service)):
        data = await read_upload(audio, svc.max_upload_bytes)
        transcript, sid, result = await run_in_threadpool(
            svc.ask_voice, data, audio.filename or "recording.webm", session_id, language)
        return VoiceAnswerOut(transcript=transcript, **_answer_fields(transcript, sid, result, include_trace))

    # ---------- images / OCR
    @router.post("/images/ocr", response_model=OcrOut, tags=["images"], summary="Extract text from an image (no ingestion)")
    async def images_ocr(image: UploadFile = File(...), languages: Optional[str] = Form(None),
                         svc: AssistantService = Depends(_service)):
        data = await read_upload(image, svc.max_upload_bytes)
        return _ocr_out(await run_in_threadpool(svc.ocr, data, image.filename or "", languages))

    @router.post("/images/ingest", response_model=ImageIngestOut, tags=["images"],
                 summary="OCR an image and add its text to the knowledge base")
    async def images_ingest(image: UploadFile = File(...), svc: AssistantService = Depends(_service)):
        data = await read_upload(image, svc.max_upload_bytes)
        res = await run_in_threadpool(svc.ingest_image, data, image.filename or "image.png")
        base = _ingest_out(res.report)
        return ImageIngestOut(**base.model_dump(), ocr_text=res.ocr_text, ocr_confidence=res.confidence)

    @router.post("/images/query", response_model=ImageAnswerOut, tags=["images"],
                 summary="Use the text in an image as the question")
    async def images_query(image: UploadFile = File(...), instruction: str = Form(""),
                           session_id: Optional[str] = Form(None), include_trace: bool = Form(False),
                           svc: AssistantService = Depends(_service)):
        data = await read_upload(image, svc.max_upload_bytes)
        ocr, sid, result = await run_in_threadpool(svc.ask_from_image, data, image.filename or "", instruction, session_id)
        return ImageAnswerOut(ocr=_ocr_out(ocr), **_answer_fields(ocr.text, sid, result, include_trace))

    # ---------- reports
    @router.post("/reports", response_model=ReportOut, tags=["reports"],
                 summary="Report Generator agent: design and write a PDF report from a conversation")
    async def create_report(body: ReportIn, svc: AssistantService = Depends(_service)):
        info = await run_in_threadpool(svc.create_report, body.session_id, body.instructions, body.title,
                                       body.question, body.last_n)
        r = info.result
        return ReportOut(report_id=info.report_id, title=r.title, pages=r.pages, exchanges=info.turns_used,
                         download_url=f"/reports/{info.report_id}", used_default_layout=r.used_fallback,
                         design_log=r.design_log)

    @router.get("/reports/{report_id}", tags=["reports"], summary="Download a generated PDF",
                response_class=FileResponse)
    def download_report(report_id: str, svc: AssistantService = Depends(_service)):
        path = svc.report_path(report_id)
        if path is None:
            raise HTTPException(404, detail="Report not found.")
        return FileResponse(path, media_type="application/pdf", filename=f"report-{report_id[:8]}.pdf")

    app.include_router(router)
    return app


app = create_app()
