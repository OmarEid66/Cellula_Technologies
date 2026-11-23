"""HTTP API tests. Skipped automatically if fastapi / httpx / python-multipart are not installed."""
import shutil

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402
from test_ocr import make_image, png_bytes  # noqa: E402

QUESTION = "What is the average CNN accuracy across the three papers?"
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed")
AUDIO = ("recording.webm", b"\x1aE\xdf\xa3" * 100, "audio/webm")


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


def test_health_and_ui_are_public(client, service):
    service.settings.api_key = "secret"
    h = client.get("/health").json()
    assert h["ready"] and h["auth_required"] and h["documents"] == 3
    assert client.get("/").status_code == 200 and "Chat with your documents" in client.get("/").text


def test_api_key_is_enforced_when_configured(client, service):
    service.settings.api_key = "secret"
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "secret"}).status_code == 200


def test_documents_upload_list_delete(client):
    r = client.post("/documents", files=[("files", ("notes.txt", b"Attention heads: 12. Context: 4096 tokens.", "text/plain")),
                                         ("files", ("bad.exe", b"MZ", "application/octet-stream"))])
    out = {x["source"]: x for x in r.json()}
    assert r.status_code == 200 and out["notes.txt"]["ok"] and not out["bad.exe"]["ok"]
    docs = client.get("/documents").json()["documents"]
    assert "notes.txt" in docs and docs["notes.txt"]["chunks"] >= 1
    assert client.delete("/documents/notes.txt").status_code == 200
    assert client.delete("/documents/notes.txt").status_code == 404


def test_query_with_trace_and_session(client):
    r = client.post("/query", json={"question": QUESTION, "include_trace": True})
    body = r.json()
    assert r.status_code == 200 and "92" in body["answer"] and body["citations"] and body["session_id"]
    assert body["trace"]["evidence_sufficient"] is True and body["trace"]["analyst_tool_calls"]
    r2 = client.post("/query", json={"question": "And the cheapest?", "session_id": body["session_id"]})
    assert r2.json()["session_id"] == body["session_id"]


def test_query_validation(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "session_id": "../bad"}).status_code == 422
    assert client.post("/query", json={}).status_code == 422


def test_voice_query(client, transcriber):
    r = client.post("/voice/query", files={"audio": AUDIO}, data={"language": "en"})
    body = r.json()
    assert r.status_code == 200 and body["transcript"] == QUESTION and "92" in body["answer"]
    assert transcriber.calls[0][2] == "en"
    assert client.post("/voice/transcribe", files={"audio": AUDIO}).json() == {"text": QUESTION}


def test_voice_rejects_bad_audio(client):
    assert client.post("/voice/query", files={"audio": ("a.txt", b"hello", "text/plain")}).status_code == 422
    assert client.post("/voice/query", files={"audio": ("a.webm", b"", "audio/webm")}).status_code == 422


def test_uploads_over_the_limit_are_refused(client, service):
    service.max_upload_bytes = 10
    assert client.post("/voice/query", files={"audio": AUDIO}).status_code == 413
    assert client.post("/documents", files=[("files", ("a.txt", b"x" * 50, "text/plain"))]).status_code == 413


@needs_tesseract
def test_image_endpoints(client):
    img = ("scan.png", png_bytes(make_image(["What is the average CNN accuracy"])), "image/png")
    ocr = client.post("/images/ocr", files={"image": img}).json()
    assert "average" in ocr["text"].lower() and ocr["confidence"] > 50
    ing = client.post("/images/ingest", files={"image": img}).json()
    assert ing["ok"] and "average" in ing["ocr_text"].lower()
    assert "scan.png" in client.get("/documents").json()["documents"]
    q = client.post("/images/query", files={"image": img}, data={"instruction": "Be brief."}).json()
    assert q["ocr"]["text"] and q["answer"] and q["question"]


def test_image_endpoints_reject_non_images(client):
    assert client.post("/images/ocr", files={"image": ("a.png", b"not an image", "image/png")}).status_code == 422
    assert client.post("/images/ocr", files={"image": ("a.pdf", b"%PDF", "application/pdf")}).status_code == 422


def test_report_flow(client):
    sid = client.post("/query", json={"question": QUESTION}).json()["session_id"]
    r = client.post("/reports", json={"session_id": sid, "instructions": "navy headings, 11pt serif", "title": "Accuracy report"})
    body = r.json()
    assert r.status_code == 200 and body["title"] == "Accuracy report" and body["pages"] >= 1 and body["exchanges"] == 1
    assert body["design_log"] and not body["used_default_layout"]
    pdf = client.get(body["download_url"])
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf" and pdf.content.startswith(b"%PDF")


def test_report_errors(client):
    assert client.post("/reports", json={"session_id": "nobody"}).status_code == 422  # nothing to report yet
    assert client.get("/reports/" + "0" * 32).status_code == 404
    assert client.get("/reports/not-a-valid-id").status_code == 404
