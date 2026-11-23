import pytest

from rag_agents.voice import AudioError, validate_audio

MB = 1024 * 1024


def test_valid_audio_keeps_a_safe_name():
    assert validate_audio(b"\x00" * 100, "../../evil/recording.webm", MB) == "recording.webm"


def test_default_name_when_missing():
    assert validate_audio(b"x", "", MB).endswith(".webm")


@pytest.mark.parametrize("data,name,match", [
    (b"", "a.webm", "empty"),
    (b"x" * (2 * MB), "a.webm", "too large"),
    (b"x", "notes.txt", "Unsupported"),
])
def test_bad_audio_is_rejected(data, name, match):
    with pytest.raises(AudioError, match=match):
        validate_audio(data, name, MB)
