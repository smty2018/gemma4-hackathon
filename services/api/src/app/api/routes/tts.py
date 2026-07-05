from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.tts import SarvamStreamingTTS, SarvamTTSRequest

router = APIRouter()


def get_tts_service() -> SarvamStreamingTTS:
    if (
        settings.sarvam_api_key is None
        or not settings.sarvam_api_key.get_secret_value().strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "sarvam_api_key_missing",
                "message": "Configure SARVAM_API_KEY to enable text-to-speech.",
            },
        )
    return SarvamStreamingTTS(settings.sarvam_api_key.get_secret_value())


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "Incrementally streamed MP3 audio.",
        },
        503: {"description": "Sarvam TTS is not configured."},
    },
)
async def stream_speech(
    request: SarvamTTSRequest,
    service: Annotated[SarvamStreamingTTS, Depends(get_tts_service)],
) -> StreamingResponse:
    return StreamingResponse(
        service.stream_audio(request),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'inline; filename="speech.mp3"',
            "X-TTS-Language": request.target_language_code,
            "X-TTS-Model": "bulbul:v3",
        },
    )
