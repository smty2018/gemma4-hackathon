from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from app.agent.orchestrator import build_analysis_plan
from app.schemas.document import AnalysisPlan

router = APIRouter()


@router.post("/analyze", response_model=AnalysisPlan, status_code=status.HTTP_202_ACCEPTED)
async def analyze_document(
    document: Annotated[UploadFile, File(description="Image or PDF to analyze")],
    language: Annotated[str, Form()] = "en",
    audience: Annotated[str, Form()] = "simple",
) -> AnalysisPlan:
    """Accept a document and return the safe processing plan.

    Model inference is deliberately not hidden behind this endpoint yet. The
    scaffold returns the typed plan that the worker will execute in the next
    milestone.
    """

    return build_analysis_plan(
        filename=document.filename or "document",
        content_type=document.content_type or "application/octet-stream",
        language=language,
        audience=audience,
    )
