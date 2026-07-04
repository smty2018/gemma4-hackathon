from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EvidenceKind(StrEnum):
    PAGE_TEXT = "page_text"
    FACT = "fact"
    DATE = "date"
    AMOUNT = "amount"


class WarningSeverity(StrEnum):
    INFO = "info"
    CAUTION = "caution"
    CRITICAL = "critical"


class ActionUrgency(StrEnum):
    IMMEDIATE = "immediate"
    SOON = "soon"
    ROUTINE = "routine"
    UNKNOWN = "unknown"


class SourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    evidence_id: str = Field(pattern=r"^[PE]\d+$")
    page: int = Field(ge=1)
    kind: EvidenceKind
    label: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=1_000)
    normalized_value: str | None = Field(default=None, max_length=100)
    currency: str | None = Field(default=None, max_length=20)
    quote: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)


class ExplanationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)


class ExplanationFact(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=1_000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0, le=1)


class RequiredAction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: str = Field(min_length=1, max_length=1_000)
    deadline: str | None = Field(default=None, max_length=200)
    urgency: ActionUrgency = ActionUrgency.UNKNOWN
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0, le=1)


class ExplanationWarning(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=1_000)
    severity: WarningSeverity = WarningSeverity.CAUTION
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class DocumentExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    language: str = Field(min_length=2, max_length=32)
    audience: str = Field(min_length=1, max_length=100)
    simple_summary: ExplanationSummary
    key_facts: list[ExplanationFact] = Field(default_factory=list, max_length=50)
    required_actions: list[RequiredAction] = Field(default_factory=list, max_length=30)
    warnings: list[ExplanationWarning] = Field(default_factory=list, max_length=50)
    source_evidence: list[SourceEvidence] = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0, le=1)
