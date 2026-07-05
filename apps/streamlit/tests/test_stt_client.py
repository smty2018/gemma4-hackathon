from typing import Any

import app as streamlit_app


class StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def test_streamlit_sends_recording_to_local_stt_api(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> StubResponse:
        calls.append({"url": url, **kwargs})
        return StubResponse(
            {
                "transcript": "আমার বিল কত?",
                "language_code": "bn-IN",
                "language_probability": 0.99,
            }
        )

    monkeypatch.setattr(streamlit_app.requests, "post", fake_post)
    streamlit_app.cached_transcription.clear()

    result = streamlit_app.cached_transcription(b"RIFFaudio", "audio/wav", "Bengali")

    assert result["transcript"] == "আমার বিল কত?"
    assert calls[0]["url"].endswith("/api/v1/stt/transcribe")
    assert calls[0]["files"] == {
        "file": ("voice-question.wav", b"RIFFaudio", "audio/wav")
    }
    assert calls[0]["data"] == {"language_code": "bn-IN"}
