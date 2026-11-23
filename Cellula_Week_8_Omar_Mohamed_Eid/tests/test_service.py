"""AssistantService: sessions, voice, images/OCR and reports (no web framework involved)."""
import io
import shutil

import pytest
from PIL import Image
from pypdf import PdfReader

from conftest import FakeTranscriber
from rag_agents.voice import AudioError
from test_ocr import make_image, png_bytes

QUESTION = "What is the average CNN accuracy across the three papers?"
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed")


def pdf_text(path) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(str(path)).pages)


def test_ask_keeps_the_conversation(service):
    sid, first = service.ask(QUESTION)
    assert "92" in first.text
    sid2, _ = service.ask("And which is cheapest?", session_id=sid)
    assert sid2 == sid
    history = service.sessions.history(sid)
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == QUESTION


@pytest.mark.parametrize("bad", ["../etc/passwd", "a b", "x" * 65, "id;drop"])
def test_session_ids_are_validated(service, bad):
    with pytest.raises(ValueError, match="session_id"):
        service.ask(QUESTION, session_id=bad)


@pytest.mark.parametrize("q", ["", "   ", "x" * 5000])
def test_questions_are_validated(service, q):
    with pytest.raises(ValueError):
        service.ask(q)


def test_voice_query_transcribes_then_answers(service, transcriber):
    transcript, sid, result = service.ask_voice(b"\x1aE\xdf\xa3" * 50, "recording.webm", language="en")
    assert transcript == QUESTION and "92" in result.text
    assert transcriber.calls == [(200, "recording.webm", "en")]
    assert service.sessions.turns(sid)[0].via == "voice"


def test_voice_rejects_empty_and_unsupported_audio(service):
    with pytest.raises(AudioError):
        service.ask_voice(b"", "recording.webm")
    with pytest.raises(AudioError):
        service.ask_voice(b"data", "notes.txt")


def test_ingest_bytes_handles_each_file_independently(service):
    reports = service.ingest_bytes([
        ("../../notes.txt", b"The model uses 12 attention heads and a 4096 token context."),
        ("virus.exe", b"MZ"),
        ("empty.txt", b""),
    ])
    by_name = {r.source: r for r in reports}
    assert by_name["notes.txt"].ok  # path components stripped
    assert "unsupported" in by_name["virus.exe"].error
    assert "empty" in by_name["empty.txt"].error
    assert "notes.txt" in service.documents()


def test_remove_and_clear_documents(service):
    assert service.remove_document("paper_a_cnn.txt") and not service.remove_document("paper_a_cnn.txt")
    service.ask(QUESTION)
    service.clear_documents()
    assert service.documents() == {} and service.sessions.turns("x") == []


@needs_tesseract
def test_ocr_and_ingest_image(service):
    img = png_bytes(make_image(["Quantum widget throughput is 4096 units"]))
    assert "widget" in service.ocr(img, "shot.png").text.lower()
    res = service.ingest_image(img, "shot.png")
    assert res.report.ok and "widget" in res.ocr_text.lower() and res.confidence > 50
    assert "shot.png" in service.documents()


@needs_tesseract
def test_image_text_becomes_the_question(service):
    ocr, sid, result = service.ask_from_image(png_bytes(make_image(["What is the average CNN accuracy"])), "q.png",
                                              extra_question="Answer briefly.")
    turn = service.sessions.turns(sid)[0]
    assert ocr.has_text and turn.via == "image"
    assert turn.question.startswith("Answer briefly.") and "average" in turn.question.lower() and result.text


@needs_tesseract
def test_image_without_text_is_an_error(service):
    with pytest.raises(ValueError, match="No text"):
        service.ask_from_image(png_bytes(Image.new("RGB", (300, 120), "white")), "blank.png")


def test_image_validation(service):
    with pytest.raises(ValueError, match="empty"):
        service.ocr(b"", "a.png")
    with pytest.raises(ValueError, match="not a supported image"):
        service.ocr(b"abc", "a.pdf")


def test_report_contains_the_conversation(service):
    sid, _ = service.ask(QUESTION)
    info = service.create_report(sid, instructions="serif fonts", title="My report")
    path = service.report_path(info.report_id)
    assert path is not None and path.read_bytes().startswith(b"%PDF")
    assert info.turns_used == 1 and info.result.title == "My report" and not info.result.used_fallback
    assert "Scripted report" in pdf_text(path)


def test_report_material_strips_the_sources_block_and_adds_citations(service):
    sid, _ = service.ask(QUESTION)
    turn = service.sessions.turns(sid)[0].to_report_turn()
    assert "**Sources**" not in turn.answer and turn.sources and turn.calculations == ["mean(92, 95, 89) = 92.0"]


def test_report_can_ask_a_new_question_first(service):
    sid, _ = service.ask(QUESTION)
    info = service.create_report(sid, question="Compare the papers' methodology.")
    assert info.turns_used == 2


def test_report_last_n(service):
    sid, _ = service.ask(QUESTION)
    service.ask("Another question", session_id=sid)
    assert service.create_report(sid, last_n=1).turns_used == 1


def test_report_needs_a_conversation(service):
    with pytest.raises(ValueError, match="Nothing to report"):
        service.create_report("unknown-session")


@pytest.mark.parametrize("bad", ["../../etc/passwd", "short", "G" * 32, "", "a" * 31 + "/"])
def test_report_path_rejects_anything_but_generated_ids(service, bad):
    assert service.report_path(bad) is None
