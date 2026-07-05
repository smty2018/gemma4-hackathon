from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.chat.pipeline import ChatGroundingError, ChatGroundingPipeline, GroundedChatRequest
from app.core.config import settings
from app.inference import GemmaAdapter
from app.schemas.chat import ChatAskRequest, GroundedChatAnswer

router = APIRouter()


@lru_cache
def get_chat_grounding_pipeline() -> ChatGroundingPipeline:
    return ChatGroundingPipeline(GemmaAdapter(model_id=settings.model_id))


@router.post("/ask", response_model=GroundedChatAnswer)
async def ask_document_question(
    request: ChatAskRequest,
    pipeline: Annotated[ChatGroundingPipeline, Depends(get_chat_grounding_pipeline)],
) -> GroundedChatAnswer:
    try:
        return await run_in_threadpool(
            pipeline.answer,
            GroundedChatRequest(
                question=request.question,
                document=request.document,
                history=request.history,
                language=request.language,
                explanation_style=request.explanation_style,
            ),
        )
    except ChatGroundingError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
                if error.retryable
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail={
                "code": error.code,
                "message": str(error),
                "retryable": error.retryable,
            },
        ) from error
