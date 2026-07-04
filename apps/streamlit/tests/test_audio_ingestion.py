from io import BytesIO
import wave

import pytest

from ingestion.audio import MAX_AUDIO_DURATION_SECONDS, inspect_audio
from ingestion.models import PreviewGenerationError


def wav_bytes(
    *, duration_seconds: float = 1.0, sample_rate: int = 8_000, channels: int = 1
) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    output = BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(channels)
        recording.setsampwidth(2)
        recording.setframerate(sample_rate)
        recording.writeframes(b"\x00\x00" * frame_count * channels)
    return output.getvalue()


def test_wav_metadata_and_playback_bytes_are_preserved() -> None:
    content = wav_bytes(duration_seconds=1.25, sample_rate=16_000, channels=1)
    preview = inspect_audio(content, extension=".wav")

    assert preview.content == content
    assert preview.content_type == "audio/wav"
    assert preview.duration_seconds == pytest.approx(1.25)
    assert preview.sample_rate == 16_000
    assert preview.channels == 1


def test_stereo_channel_count_is_reported() -> None:
    preview = inspect_audio(wav_bytes(channels=2), extension=".WAV")

    assert preview.channels == 2


def test_audio_duration_limit_is_enforced() -> None:
    content = wav_bytes(
        duration_seconds=MAX_AUDIO_DURATION_SECONDS + 1,
        sample_rate=1_000,
    )

    with pytest.raises(PreviewGenerationError) as error:
        inspect_audio(content, extension=".wav")

    assert error.value.code == "audio_too_long"


def test_audio_at_gemma4_duration_limit_is_accepted() -> None:
    preview = inspect_audio(
        wav_bytes(
            duration_seconds=MAX_AUDIO_DURATION_SECONDS,
            sample_rate=1_000,
        ),
        extension=".wav",
    )

    assert preview.duration_seconds == pytest.approx(30)


def test_corrupt_audio_has_stable_error_code() -> None:
    with pytest.raises(PreviewGenerationError) as error:
        inspect_audio(b"RIFF\x00\x00\x00\x00WAVEnot playable", extension=".wav")

    assert error.value.code == "invalid_audio"


def test_unknown_audio_extension_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported audio extension"):
        inspect_audio(wav_bytes(), extension=".ogg")
