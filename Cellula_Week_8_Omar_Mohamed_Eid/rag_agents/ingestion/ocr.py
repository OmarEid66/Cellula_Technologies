"""OCR: extract text from images with Tesseract (through pytesseract).

The text is returned as plain text so it can go through the same Loader -> Parser -> Cleaning -> Chunking ->
Embeddings -> Vector DB pipeline as any other document, or be used directly as a question.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

_LANG_RE = re.compile(r"^[A-Za-z_]{2,12}(\+[A-Za-z_]{2,12})*$")
_TARGET_MIN_SIDE = 1400  # small screenshots OCR much better when upscaled
_MAX_SIDE = 4500


class OcrUnavailableError(RuntimeError):
    """Tesseract (or pytesseract / Pillow) is not installed."""


class OcrError(ValueError):
    """The file is not a readable image, or the OCR language is invalid."""


@dataclass
class OcrResult:
    text: str
    confidence: float  # mean word confidence, 0-100 (0 when no words were found)
    word_count: int
    width: int
    height: int
    language: str
    warnings: list[str] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


def is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def _preprocess(img):
    """Orientation fix, grayscale, upscale small images, stretch contrast."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):  # flatten transparency onto white (transparent PNGs become black otherwise)
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = ImageOps.grayscale(img)
    w, h = img.size
    short = min(w, h)
    if short < _TARGET_MIN_SIDE:
        scale = min(_TARGET_MIN_SIDE / short, 3.0)
        if max(w, h) * scale > _MAX_SIDE:
            scale = _MAX_SIDE / max(w, h)
        if scale > 1.05:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(img, cutoff=1)


def _open_image(source: str | Path | bytes | BinaryIO):
    from PIL import Image, UnidentifiedImageError

    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, (str, Path)):
            img = Image.open(source)
        else:
            img = Image.open(source)
        img.load()
        if getattr(img, "n_frames", 1) > 1:  # animated GIF / multi-page TIFF: first frame only
            img.seek(0)
        return img
    except UnidentifiedImageError as exc:
        raise OcrError("The file is not a readable image.") from exc
    except Image.DecompressionBombError as exc:
        raise OcrError("The image is too large to process.") from exc
    except OSError as exc:
        raise OcrError(f"The image could not be read: {exc}") from exc


def ocr_image(
    source: str | Path | bytes | BinaryIO, languages: str = "eng", min_confidence: int = 50
) -> OcrResult:
    """Run OCR on an image and rebuild the text with its line and paragraph structure."""
    if not _LANG_RE.match(languages or ""):
        raise OcrError(f"Invalid OCR language string '{languages}'. Use tesseract codes like 'eng' or 'eng+ara'.")
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise OcrUnavailableError("OCR needs 'pytesseract' and 'Pillow' (pip install pytesseract pillow).") from exc

    img = _preprocess(_open_image(source))
    try:
        data = pytesseract.image_to_data(
            img, lang=languages, config="--psm 3", output_type=pytesseract.Output.DICT, timeout=120
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrUnavailableError(
            "The Tesseract OCR engine is not installed. Install it (Windows: UB-Mannheim installer, "
            "macOS: 'brew install tesseract', Ubuntu: 'apt install tesseract-ocr') and make sure it is on PATH."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise OcrError(f"Tesseract failed (is the language '{languages}' installed?): {exc}") from exc
    except RuntimeError as exc:  # pytesseract raises RuntimeError('Tesseract process timeout')
        raise OcrError(f"OCR timed out: {exc}") from exc

    lines: dict[tuple[int, int, int], list[str]] = {}
    order: list[tuple[int, int, int]] = []
    confs: list[float] = []
    for i, raw in enumerate(data["text"]):
        word = (raw or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(word)
        confs.append(conf)

    out: list[str] = []
    prev_par: tuple[int, int] | None = None
    for key in order:
        par = key[:2]
        if prev_par is not None and par != prev_par:
            out.append("")  # blank line between paragraphs / blocks
        out.append(" ".join(lines[key]))
        prev_par = par
    text = "\n".join(out).strip()

    mean_conf = sum(confs) / len(confs) if confs else 0.0
    warnings: list[str] = []
    if not text:
        warnings.append("no text was detected in the image")
    elif mean_conf < min_confidence:
        warnings.append(f"low OCR confidence ({mean_conf:.0f}/100): the extracted text may contain errors")
    return OcrResult(
        text=text,
        confidence=round(mean_conf, 1),
        word_count=len(confs),
        width=img.size[0],
        height=img.size[1],
        language=languages,
        warnings=warnings,
    )
