from io import BytesIO

import mutagen
from mutagen import MutagenError

from ingestion.models import AudioPreview, PreviewGenerationError


MAX_AUDIO_DURATION_SECONDS = 60

AUDIO_MIMES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
}


def _optional_integer(info: object, attribute: str) -> int | None:
    value = getattr(info, attribute, None)
    return int(value) if value is not None else None


def inspect_audio(content: bytes, *, extension: str) -> AudioPreview:
    normalized_extension = extension.lower()
    content_type = AUDIO_MIMES.get(normalized_extension)
    if content_type is None:
        raise ValueError(f"Unsupported audio extension: {extension}")

    try:
        audio = mutagen.File(BytesIO(content))
        if audio is None or getattr(audio, "info", None) is None:
            raise PreviewGenerationError(
                "invalid_audio",
                "The audio file could not be decoded.",
            )

        duration = float(audio.info.length)
        if duration <= 0:
            raise PreviewGenerationError(
                "audio_has_no_duration",
                "The audio file does not contain a playable recording.",
            )
        if duration > MAX_AUDIO_DURATION_SECONDS:
            raise PreviewGenerationError(
                "audio_too_long",
                f"Audio recordings must be {MAX_AUDIO_DURATION_SECONDS} seconds or shorter.",
            )

        return AudioPreview(
            content=content,
            content_type=content_type,
            duration_seconds=duration,
            sample_rate=_optional_integer(audio.info, "sample_rate"),
            channels=_optional_integer(audio.info, "channels"),
        )
    except PreviewGenerationError:
        raise
    except (MutagenError, EOFError, OSError, ValueError) as error:
        raise PreviewGenerationError(
            "invalid_audio",
            "The audio file could not be decoded.",
        ) from error
