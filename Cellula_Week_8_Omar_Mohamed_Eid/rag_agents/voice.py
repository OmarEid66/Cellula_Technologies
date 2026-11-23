"""Voice queries: speech-to-text for audio recorded in the browser (or uploaded) so it can be asked like typed text."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

from .config import Settings

AUDIO_EXTENSIONS = {".webm", ".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".ogg", ".oga", ".flac"}


class SpeechUnavailableError(RuntimeError):
    """The speech-to-text backend cannot be used (missing package or API key)."""


class AudioError(ValueError):
    """The audio is empty, unsupported, or no speech could be recognised."""


class Transcriber(Protocol):
    def transcribe(self, audio: bytes, filename: str, language: str | None = None) -> str: ...


def validate_audio(audio: bytes, filename: str, max_bytes: int) -> str:
    """Returns a safe file name (with extension) for the transcription request; raises AudioError when unusable."""
    if not audio:
        raise AudioError("The audio recording is empty.")
    if len(audio) > max_bytes:
        raise AudioError(f"The audio file is too large (limit {max_bytes // (1024 * 1024)} MB).")
    name = Path(filename or "recording.webm").name or "recording.webm"
    if Path(name).suffix.lower() not in AUDIO_EXTENSIONS:
        raise AudioError(f"Unsupported audio type '{Path(name).suffix}'. Supported: {sorted(AUDIO_EXTENSIONS)}")
    return name


class OpenAITranscriber:
    """Speech-to-text through the OpenAI transcription API (whisper-1 by default)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - openai ships with langchain-openai
                raise SpeechUnavailableError("The 'openai' package is required for voice queries.") from exc
            import os

            if not os.getenv("OPENAI_API_KEY"):
                raise SpeechUnavailableError("OPENAI_API_KEY is not set, so speech cannot be transcribed.")
            self._client = OpenAI()
        return self._client

    def transcribe(self, audio: bytes, filename: str, language: str | None = None) -> str:
        client = self._get_client()
        kwargs = {"model": self.settings.stt_model, "file": (filename, io.BytesIO(audio))}
        lang = language or self.settings.stt_language
        if lang:
            kwargs["language"] = lang
        try:
            result = client.audio.transcriptions.create(**kwargs)
        except Exception as exc:  # network, bad audio, quota ... surface a clean message
            raise AudioError(f"Transcription failed: {type(exc).__name__}: {exc}") from exc
        text = (getattr(result, "text", None) or (result if isinstance(result, str) else "")).strip()
        if not text:
            raise AudioError("No speech was recognised in the recording.")
        return text
