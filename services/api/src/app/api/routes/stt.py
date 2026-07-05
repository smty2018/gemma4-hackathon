from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.stt import (
    SarvamSpeechToText,
    SarvamSTTError,
    SarvamSTTInputError,
    SarvamSTTResult,
    STTLanguageCode,
)

router = APIRouter()
MAX_STT_BYTES = 15 * 1024 * 1024


def get_stt_service() -> SarvamSpeechToText:
    if (
        settings.sarvam_api_key is None
        or not settings.sarvam_api_key.get_secret_value().strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "sarvam_api_key_missing",
                "message": "Configure SARVAM_API_KEY to enable speech-to-text.",
            },
        )
    return SarvamSpeechToText(settings.sarvam_api_key.get_secret_value())


@router.post("/transcribe", response_model=SarvamSTTResult)
async def transcribe_speech(
    file: Annotated[UploadFile, File(description="A recording of up to 30 seconds.")],
    service: Annotated[SarvamSpeechToText, Depends(get_stt_service)],
    language_code: Annotated[STTLanguageCode, Form()] = "unknown",
) -> SarvamSTTResult:
    audio = await file.read(MAX_STT_BYTES + 1)
    if len(audio) > MAX_STT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "audio_too_large", "message": "Audio must be 15 MB or smaller."},
        )
    try:
        return await run_in_threadpool(
            service.transcribe,
            audio,
            filename=file.filename or "voice-question.wav",
            content_type=file.content_type or "audio/wav",
            language_code=language_code,
        )
    except SarvamSTTInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": error.code, "message": str(error)},
        ) from error
    except SarvamSTTError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": error.code, "message": str(error)},
        ) from error
