from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes.stt import get_stt_service
from app.main import app
from app.stt import (
    SarvamSpeechToText,
    SarvamSTTError,
    SarvamSTTInputError,
    SarvamSTTResponseError,
    SarvamSTTResult,
)


class FakeSpeechToTextResource:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.speech_to_text = FakeSpeechToTextResource(response)


@pytest.mark.parametrize("language_code", ["unknown", "en-IN", "hi-IN", "bn-IN"])
def test_saaras_v3_transcribes_supported_ui_languages(language_code: str) -> None:
    response = SimpleNamespace(
        transcript="नमस्ते",
        language_code="hi-IN",
        language_probability=0.97,
        request_id="request-1",
    )
    client = FakeClient(response)
    service = SarvamSpeechToText("secret", client_factory=lambda _key: client)

    result = service.transcribe(
        b"RIFFaudio",
        filename="question.wav",
        language_code=language_code,  # type: ignore[arg-type]
    )

    assert result.transcript == "नमस्ते"
    assert result.language_code == "hi-IN"
    assert result.language_probability == 0.97
    assert client.speech_to_text.calls == [
        {
            "file": ("question.wav", b"RIFFaudio", "audio/wav"),
            "model": "saaras:v3",
            "mode": "transcribe",
            "language_code": language_code,
        }
    ]


def test_service_rejects_missing_key_and_empty_audio() -> None:
    with pytest.raises(SarvamSTTError, match="SARVAM_API_KEY"):
        SarvamSpeechToText("  ")

    service = SarvamSpeechToText("secret", client_factory=lambda _key: FakeClient(None))
    with pytest.raises(SarvamSTTInputError) as error:
        service.transcribe(b"")
    assert error.value.code == "empty_audio"


def test_service_rejects_empty_transcript() -> None:
    client = FakeClient(SimpleNamespace(transcript="   "))
    service = SarvamSpeechToText("secret", client_factory=lambda _key: client)

    with pytest.raises(SarvamSTTResponseError) as error:
        service.transcribe(b"RIFFaudio")

    assert error.value.code == "empty_transcript"


def test_api_endpoint_returns_transcript() -> None:
    class StubService:
        def transcribe(self, _audio: bytes, **_kwargs: Any) -> SarvamSTTResult:
            return SarvamSTTResult(
                transcript="আমার বিল কত?",
                language_code="bn-IN",
                language_probability=0.98,
                request_id="request-2",
            )

    app.dependency_overrides[get_stt_service] = lambda: StubService()
    try:
        response = TestClient(app).post(
            "/api/v1/stt/transcribe",
            files={"file": ("question.wav", b"RIFFaudio", "audio/wav")},
            data={"language_code": "bn-IN"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "transcript": "আমার বিল কত?",
        "language_code": "bn-IN",
        "language_probability": 0.98,
        "request_id": "request-2",
        "model": "saaras:v3",
    }


def test_api_endpoint_rejects_empty_audio() -> None:
    class StubService:
        def transcribe(self, _audio: bytes, **_kwargs: Any) -> SarvamSTTResult:
            raise SarvamSTTInputError("empty_audio", "Provide audio.")

    app.dependency_overrides[get_stt_service] = lambda: StubService()
    try:
        response = TestClient(app).post(
            "/api/v1/stt/transcribe",
            files={"file": ("question.wav", b"", "audio/wav")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "empty_audio"
