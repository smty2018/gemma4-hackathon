from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.agent.orchestrator import build_analysis_plan
from app.core.config import settings
from app.documents import MAX_DOCUMENT_BYTES, DocumentAnalysisService, DocumentIngestionError
from app.explanation.pipeline import ExplanationPipelineError
from app.inference import GemmaAdapter
from app.ocr.pipeline import OcrPipelineError
from app.schemas.document import AnalysisPlan, DocumentAnalysisResult

router = APIRouter()


@lru_cache
def get_document_analysis_service() -> DocumentAnalysisService:
    return DocumentAnalysisService(GemmaAdapter(model_id=settings.model_id))


@router.post("/plan", response_model=AnalysisPlan, status_code=status.HTTP_202_ACCEPTED)
async def plan_document_analysis(
    document: Annotated[UploadFile, File(description="Image or PDF to analyze")],
    language: Annotated[str, Form()] = "English",
    audience: Annotated[str, Form()] = "general public",
) -> AnalysisPlan:
    return build_analysis_plan(
        filename=document.filename or "document",
        content_type=document.content_type or "application/octet-stream",
        language=language,
        audience=audience,
    )


@router.post("/analyze", response_model=DocumentAnalysisResult)
async def analyze_document(
    document: Annotated[UploadFile, File(description="Image or PDF to analyze")],
    service: Annotated[
        DocumentAnalysisService,
        Depends(get_document_analysis_service),
    ],
    language: Annotated[str, Form()] = "English",
    audience: Annotated[str, Form()] = "general public",
) -> DocumentAnalysisResult:
    content = await document.read(MAX_DOCUMENT_BYTES + 1)
    try:
        return await run_in_threadpool(
            service.analyze,
            filename=document.filename or "document",
            content_type=document.content_type,
            content=content,
            language=language,
            audience=audience,
        )
    except DocumentIngestionError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"code": error.code, "message": str(error)},
        ) from error
    except (OcrPipelineError, ExplanationPipelineError) as error:
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
