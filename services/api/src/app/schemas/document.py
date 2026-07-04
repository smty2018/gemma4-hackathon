from enum import StrEnum

from pydantic import BaseModel, Field


class AnalysisStage(StrEnum):
    ACCEPTED = "accepted"
    EXTRACT_EVIDENCE = "extract_evidence"
    VERIFY_FACTS = "verify_facts"
    BUILD_ACTION_PLAN = "build_action_plan"


class PlannedStep(BaseModel):
    stage: AnalysisStage
    description: str


class AnalysisPlan(BaseModel):
    document_name: str
    content_type: str
    language: str = Field(min_length=2, max_length=16)
    audience: str
    status: AnalysisStage = AnalysisStage.ACCEPTED
    steps: list[PlannedStep]
    requires_confirmation_before_actions: bool = True


class EvidenceFact(BaseModel):
    label: str
    value: str
    page: int | None = None
    evidence_text: str
    confidence: float = Field(ge=0, le=1)
