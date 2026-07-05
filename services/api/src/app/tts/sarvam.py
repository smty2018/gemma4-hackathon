from __future__ import annotations

import base64
import binascii
import re
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_MODEL = "bulbul:v3"
SUPPORTED_LANGUAGE_CODES = frozenset({"en-IN", "hi-IN", "bn-IN"})
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥])\s+")


class SarvamTTSError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SarvamTTSResponseError(SarvamTTSError):
    pass


class SarvamTTSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=2500)
    target_language_code: Literal["en-IN", "hi-IN", "bn-IN"] = "en-IN"
    speaker: str = Field(default="shubh", min_length=1, max_length=50)
    pace: float = Field(default=1.0, ge=0.5, le=2.0)
    min_buffer_size: int = Field(default=50, ge=30, le=200)
    max_chunk_length: int = Field(default=200, ge=50, le=500)
    speech_sample_rate: Literal[8000, 16000, 22050, 24000] = 24000
    output_audio_bitrate: Literal["32k", "64k", "96k", "128k", "192k"] = "128k"

    @field_validator("speaker")
    @classmethod
    def speaker_must_be_lowercase(cls, value: str) -> str:
        if value != value.lower():
            raise ValueError("speaker must be lowercase")
        return value

    @model_validator(mode="after")
    def buffer_must_fit_chunk(self) -> SarvamTTSRequest:
        if self.min_buffer_size > self.max_chunk_length:
            raise ValueError("min_buffer_size cannot exceed max_chunk_length")
        return self


def _default_client_factory(api_key: str) -> Any:
    try:
        from sarvamai import AsyncSarvamAI
    except ImportError as error:  # pragma: no cover - depends on optional runtime installation
        raise SarvamTTSError(
            "sarvam_dependency_missing",
            "Install the API package dependencies to enable Sarvam TTS.",
        ) from error
    return AsyncSarvamAI(api_subscription_key=api_key)


class SarvamStreamingTTS:
    def __init__(
        self,
        api_key: str,
        *,
        client_factory: Callable[[str], Any] = _default_client_factory,
    ) -> None:
        if not api_key.strip():
            raise SarvamTTSError(
                "sarvam_api_key_missing",
                "SARVAM_API_KEY is required for text-to-speech.",
            )
        self._api_key = api_key
        self._client_factory = client_factory

    async def stream_audio(self, request: SarvamTTSRequest) -> AsyncIterator[bytes]:
        client = self._client_factory(self._api_key)
        try:
            connection = client.text_to_speech_streaming.connect(
                model=DEFAULT_MODEL,
                send_completion_event=True,
            )
            async with connection as websocket:
                await websocket.configure(
                    target_language_code=request.target_language_code,
                    speaker=request.speaker,
                    pace=request.pace,
                    min_buffer_size=request.min_buffer_size,
                    max_chunk_length=request.max_chunk_length,
                    speech_sample_rate=request.speech_sample_rate,
                    output_audio_codec="mp3",
                    output_audio_bitrate=request.output_audio_bitrate,
                )
                for text_chunk in sentence_chunks(request.text, request.max_chunk_length):
                    await websocket.convert(text_chunk)
                await websocket.flush()

                received_final_event = False
                async for message in websocket:
                    audio = _message_value(message, "audio")
                    if audio:
                        yield _decode_audio(audio)

                    event_type = _message_value(message, "event_type")
                    if event_type == "final":
                        received_final_event = True
                        break

                    if getattr(message, "type", None) == "error":
                        detail = _message_value(message, "message") or "Sarvam rejected the stream."
                        raise SarvamTTSResponseError("sarvam_stream_error", str(detail))

                if not received_final_event:
                    raise SarvamTTSResponseError(
                        "sarvam_stream_incomplete",
                        "Sarvam closed the stream before the final event.",
                    )
        except SarvamTTSError:
            raise
        except Exception as error:
            raise SarvamTTSError(
                "sarvam_connection_error",
                "Sarvam text-to-speech streaming failed.",
            ) from error


def sentence_chunks(text: str, max_length: int) -> Iterable[str]:
    paragraphs = (part.strip() for part in _SENTENCE_BOUNDARY.split(text.strip()))
    for paragraph in paragraphs:
        if not paragraph:
            continue
        remaining = paragraph
        while len(remaining) > max_length:
            split_at = remaining.rfind(" ", 0, max_length + 1)
            if split_at <= 0:
                split_at = max_length
            yield remaining[:split_at].strip()
            remaining = remaining[split_at:].strip()
        if remaining:
            yield remaining


def _message_value(message: Any, field: str) -> Any:
    data = getattr(message, "data", None)
    if isinstance(data, dict):
        return data.get(field)
    return getattr(data, field, None)


def _decode_audio(encoded_audio: str) -> bytes:
    try:
        decoded = base64.b64decode(encoded_audio, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SarvamTTSResponseError(
            "invalid_sarvam_audio",
            "Sarvam returned an invalid audio chunk.",
        ) from error
    if not decoded:
        raise SarvamTTSResponseError(
            "empty_sarvam_audio",
            "Sarvam returned an empty audio chunk.",
        )
    return decoded
