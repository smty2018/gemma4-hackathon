from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = "saaras:v3"
STTLanguageCode = Literal["unknown", "en-IN", "hi-IN", "bn-IN"]


class SarvamSTTError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SarvamSTTInputError(SarvamSTTError):
    pass


class SarvamSTTResponseError(SarvamSTTError):
    pass


class SarvamSTTResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transcript: str = Field(min_length=1)
    language_code: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    request_id: str | None = None
    model: str = DEFAULT_MODEL


def _default_client_factory(api_key: str) -> Any:
    try:
        from sarvamai import SarvamAI
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise SarvamSTTError(
            "sarvam_dependency_missing",
            "Install the API package dependencies to enable Sarvam STT.",
        ) from error
    return SarvamAI(api_subscription_key=api_key)


class SarvamSpeechToText:
    def __init__(
        self,
        api_key: str,
        *,
        client_factory: Callable[[str], Any] = _default_client_factory,
    ) -> None:
        if not api_key.strip():
            raise SarvamSTTError(
                "sarvam_api_key_missing",
                "SARVAM_API_KEY is required for speech-to-text.",
            )
        self._api_key = api_key
        self._client_factory = client_factory

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "voice-question.wav",
        content_type: str = "audio/wav",
        language_code: STTLanguageCode = "unknown",
    ) -> SarvamSTTResult:
        if not audio:
            raise SarvamSTTInputError(
                "empty_audio",
                "Provide a non-empty audio recording.",
            )

        safe_filename = Path(filename).name or "voice-question.wav"
        client = self._client_factory(self._api_key)
        try:
            response = client.speech_to_text.transcribe(
                file=(safe_filename, audio, content_type),
                model=DEFAULT_MODEL,
                mode="transcribe",
                language_code=language_code,
            )
        except SarvamSTTError:
            raise
        except Exception as error:
            raise SarvamSTTError(
                "sarvam_connection_error",
                "Sarvam speech-to-text failed.",
            ) from error

        transcript = str(getattr(response, "transcript", "")).strip()
        if not transcript:
            raise SarvamSTTResponseError(
                "empty_transcript",
                "Sarvam returned an empty transcript.",
            )

        return SarvamSTTResult(
            transcript=transcript,
            language_code=getattr(response, "language_code", None),
            language_probability=getattr(response, "language_probability", None),
            request_id=getattr(response, "request_id", None),
        )
