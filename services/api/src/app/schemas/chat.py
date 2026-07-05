from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.explanation import DocumentExplanation, SourceEvidence


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: ChatRole
    content: str = Field(min_length=1, max_length=4_000)


class ToolResultEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    result_id: str = Field(pattern=r"^T\d+$")
    tool_name: str = Field(min_length=1, max_length=64)
    summary: str = Field(min_length=1, max_length=1_000)
    data: dict[str, Any] = Field(default_factory=dict)
    succeeded: bool
    executed_at: datetime


class GroundedChatAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=4_000)
    answerable: bool
    document_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    tool_result_ids: list[str] = Field(default_factory=list, max_length=20)
    document_sources: list[SourceEvidence] = Field(default_factory=list, max_length=20)
    tool_sources: list[ToolResultEvidence] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0, le=1)


class ChatAskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2_000)
    document: DocumentExplanation
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    language: str = Field(default="English", min_length=2, max_length=32)
    explanation_style: str = Field(default="Simple", min_length=1, max_length=100)
