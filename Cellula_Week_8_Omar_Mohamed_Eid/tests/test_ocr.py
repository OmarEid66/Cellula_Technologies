"""OCR + image ingestion (needs the Tesseract binary; skipped when it is not installed)."""
import io
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from rag_agents.ingestion.loader import SUPPORTED_EXTENSIONS, load_file
from rag_agents.ingestion.ocr import OcrError, ocr_image

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed")


def make_image(lines: list[str], size=(900, 220), scale: int = 34) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=scale)
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((30, 20 + i * (scale + 14)), line, fill="black", font=font)
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_images_are_a_supported_upload_type():
    assert {".png", ".jpg", ".jpeg", ".webp"} <= SUPPORTED_EXTENSIONS


@needs_tesseract
def test_ocr_reads_text_and_reports_confidence():
    res = ocr_image(png_bytes(make_image(["Average CNN accuracy is 92 percent", "Paper A beats Paper C"])))
    text = res.text.lower()
    assert "accuracy" in text and "92" in text and "paper" in text
    assert res.has_text and res.word_count >= 8 and res.confidence > 50


@needs_tesseract
def test_ocr_blank_image_warns_instead_of_failing():
    res = ocr_image(png_bytes(Image.new("RGB", (300, 120), "white")))
    assert not res.has_text and any("no text" in w for w in res.warnings)


@needs_tesseract
def test_transparent_png_is_flattened_onto_white():
    img = make_image(["Transparent background works"]).convert("RGBA")
    img.putalpha(img.convert("L").point(lambda v: 0 if v > 250 else 255))  # white -> fully transparent
    assert "transparent" in ocr_image(png_bytes(img)).text.lower()


def test_not_an_image_is_rejected():
    with pytest.raises(OcrError):
        ocr_image(b"this is not an image")


def test_invalid_language_string_is_rejected():
    with pytest.raises(OcrError):
        ocr_image(png_bytes(Image.new("RGB", (10, 10))), languages="eng; rm -rf /")


@needs_tesseract
def test_loader_turns_an_image_into_a_document(tmp_path):
    path = tmp_path / "scan.png"
    path.write_bytes(png_bytes(make_image(["Learning rate 0.001 and batch size 64"])))
    loaded = load_file(path)
    assert loaded.doc_type == "image" and loaded.source == "scan.png"
    assert len(loaded.pages) == 1 and "batch" in loaded.pages[0].text.lower()
    assert loaded.ocr_confidence and loaded.ocr_confidence > 50


@needs_tesseract
def test_image_without_text_loads_with_a_warning(tmp_path):
    path = tmp_path / "blank.png"
    path.write_bytes(png_bytes(Image.new("RGB", (200, 100), "white")))
    loaded = load_file(path)
    assert loaded.pages == [] and loaded.warnings


@needs_tesseract
def test_image_is_ingested_into_the_knowledge_base(kb, settings, tmp_path):
    from rag_agents.ingestion import ingest_file

    path = tmp_path / "note.png"
    path.write_bytes(png_bytes(make_image(["The transformer has twelve attention heads"])))
    report = ingest_file(path, kb, settings)
    assert report.ok and report.text_chunks >= 1
    assert "note.png" in kb.list_sources()
