from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class OcrSource(StrEnum):
    IMAGE = "image"
    EMBEDDED_TEXT = "embedded_text"
    IMAGE_AND_EMBEDDED_TEXT = "image_and_embedded_text"


class OcrEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    page: int = Field(ge=1)
    label: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=1_000)
    evidence_text: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    grounded: bool


class OcrDate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    page: int = Field(ge=1)
    value: str = Field(min_length=1, max_length=200)
    normalized_value: str | None = Field(default=None, max_length=40)
    evidence_text: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    grounded: bool


class OcrAmount(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    page: int = Field(ge=1)
    value: str = Field(min_length=1, max_length=200)
    normalized_value: str | None = Field(default=None, max_length=100)
    currency: str | None = Field(default=None, max_length=20)
    evidence_text: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    grounded: bool


class OcrPageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    source: OcrSource
    text: str = Field(max_length=100_000)
    evidence: list[OcrEvidence] = Field(default_factory=list)
    dates: list[OcrDate] = Field(default_factory=list)
    amounts: list[OcrAmount] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class OcrDocumentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_count: int = Field(ge=1)
    text: str
    pages: list[OcrPageResult]
    evidence: list[OcrEvidence] = Field(default_factory=list)
    dates: list[OcrDate] = Field(default_factory=list)
    amounts: list[OcrAmount] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
