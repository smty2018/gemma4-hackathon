import asyncio
import base64
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes.tts import get_tts_service
from app.main import app
from app.tts.sarvam import (
    SarvamStreamingTTS,
    SarvamTTSError,
    SarvamTTSRequest,
    SarvamTTSResponseError,
    sentence_chunks,
)


class FakeSocket:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.configure_calls: list[dict[str, Any]] = []
        self.convert_calls: list[str] = []
        self.flush_count = 0

    async def configure(self, **kwargs: Any) -> None:
        self.configure_calls.append(kwargs)

    async def convert(self, text: str) -> None:
        self.convert_calls.append(text)

    async def flush(self) -> None:
        self.flush_count += 1

    def __aiter__(self) -> "FakeSocket":
        self._iterator = iter(self.messages)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iterator)
        except StopIteration as error:
            raise StopAsyncIteration from error


class FakeConnection:
    def __init__(self, socket: FakeSocket) -> None:
        self.socket = socket

    async def __aenter__(self) -> FakeSocket:
        return self.socket

    async def __aexit__(self, *_args: Any) -> None:
        return None


class FakeStreamingResource:
    def __init__(self, socket: FakeSocket) -> None:
        self.socket = socket
        self.connect_calls: list[dict[str, Any]] = []

    def connect(self, **kwargs: Any) -> FakeConnection:
        self.connect_calls.append(kwargs)
        return FakeConnection(self.socket)


class FakeClient:
    def __init__(self, socket: FakeSocket) -> None:
        self.text_to_speech_streaming = FakeStreamingResource(socket)


def audio_message(content: bytes) -> Any:
    return SimpleNamespace(
        type="audio",
        data=SimpleNamespace(audio=base64.b64encode(content).decode("ascii")),
    )


def final_message() -> Any:
    return SimpleNamespace(type="event", data=SimpleNamespace(event_type="final"))


async def collect_audio(service: SarvamStreamingTTS, request: SarvamTTSRequest) -> bytes:
    return b"".join([chunk async for chunk in service.stream_audio(request)])


@pytest.mark.parametrize(
    ("language_code", "text"),
    [
        ("en-IN", "Your document is ready."),
        ("hi-IN", "आपका दस्तावेज़ तैयार है।"),
        ("bn-IN", "আপনার নথি প্রস্তুত।"),
    ],
)
def test_bulbul_v3_streams_supported_languages(language_code: str, text: str) -> None:
    socket = FakeSocket([audio_message(b"first"), audio_message(b"second"), final_message()])
    client = FakeClient(socket)
    service = SarvamStreamingTTS("secret", client_factory=lambda _key: client)

    audio = asyncio.run(
        collect_audio(
            service,
            SarvamTTSRequest(text=text, target_language_code=language_code),
        )
    )

    assert audio == b"firstsecond"
    assert client.text_to_speech_streaming.connect_calls == [
        {"model": "bulbul:v3", "send_completion_event": True}
    ]
    assert socket.configure_calls[0]["target_language_code"] == language_code
    assert socket.configure_calls[0]["output_audio_codec"] == "mp3"
    assert socket.configure_calls[0]["speech_sample_rate"] == 24000
    assert socket.convert_calls == [text]
    assert socket.flush_count == 1


def test_sentence_chunking_preserves_content_and_limits_chunk_size() -> None:
    text = "First sentence. " + "word " * 70 + "শেষ বাক্য।"

    chunks = list(sentence_chunks(text, max_length=80))

    assert chunks[0] == "First sentence."
    assert chunks[-1].endswith("শেষ বাক্য।")
    assert all(0 < len(chunk) <= 80 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ") == text.strip().replace("  ", " ")


def test_request_rejects_unsupported_language_and_invalid_buffer() -> None:
    with pytest.raises(ValidationError):
        SarvamTTSRequest(text="Bonjour", target_language_code="fr-FR")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="min_buffer_size cannot exceed"):
        SarvamTTSRequest(text="Hello", min_buffer_size=200, max_chunk_length=100)


def test_missing_key_fails_before_opening_a_connection() -> None:
    with pytest.raises(SarvamTTSError) as error:
        SarvamStreamingTTS("  ")

    assert error.value.code == "sarvam_api_key_missing"


def test_invalid_audio_chunk_has_stable_error_code() -> None:
    socket = FakeSocket(
        [SimpleNamespace(type="audio", data=SimpleNamespace(audio="not-base64")), final_message()]
    )
    service = SarvamStreamingTTS(
        "secret",
        client_factory=lambda _key: FakeClient(socket),
    )

    with pytest.raises(SarvamTTSResponseError) as error:
        asyncio.run(collect_audio(service, SarvamTTSRequest(text="Hello")))

    assert error.value.code == "invalid_sarvam_audio"


def test_api_endpoint_proxies_streamed_audio() -> None:
    class StubService:
        async def stream_audio(self, _request: SarvamTTSRequest):
            yield b"ID3"
            yield b"audio"

    app.dependency_overrides[get_tts_service] = lambda: StubService()
    try:
        response = TestClient(app).post(
            "/api/v1/tts/stream",
            json={
                "text": "আপনার নথি প্রস্তুত।",
                "target_language_code": "bn-IN",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.headers["x-tts-language"] == "bn-IN"
    assert response.headers["x-tts-model"] == "bulbul:v3"
    assert response.content == b"ID3audio"
