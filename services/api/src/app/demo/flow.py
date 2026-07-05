from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from app.chat.pipeline import ChatGroundingPipeline, GroundedChatRequest
from app.explanation.pipeline import ExplanationPipeline, ExplanationRequest
from app.inference.gemma import GemmaRequest, GemmaResponse
from app.ocr.pipeline import OcrPageInput, OcrPipeline
from app.schemas.chat import ChatRole, ChatTurn, GroundedChatAnswer
from app.schemas.explanation import DocumentExplanation
from app.schemas.ocr import OcrDocumentResult
from app.tools.executor import ToolExecutionReceipt


@dataclass(frozen=True)
class DemoFlowRequest:
    document_text: str
    language: str = "English"
    audience: str = "general public"
    explanation_style: str = "Simple"
    follow_up_question: str = "What should I do and by when?"
    tool_results: Sequence[ToolExecutionReceipt] = ()


class DemoFlowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ocr: OcrDocumentResult
    explanation: DocumentExplanation
    follow_up: GroundedChatAnswer


class GemmaGenerator(Protocol):
    def generate(self, request: GemmaRequest) -> GemmaResponse: ...


class DemoFlow:
    def __init__(self, gemma: GemmaGenerator) -> None:
        self._gemma = gemma

    def run(self, request: DemoFlowRequest) -> DemoFlowResult:
        ocr = OcrPipeline(self._gemma).extract(
            [OcrPageInput(page=1, embedded_text=request.document_text)]
        )
        explanation = ExplanationPipeline(self._gemma).explain(
            ExplanationRequest(
                ocr=ocr,
                language=request.language,
                audience=request.audience,
            )
        )
        history = [
            ChatTurn(role=ChatRole.USER, content="Explain this document."),
            ChatTurn(
                role=ChatRole.ASSISTANT,
                content=explanation.simple_summary.text,
            ),
        ]
        follow_up = ChatGroundingPipeline(self._gemma).answer(
            GroundedChatRequest(
                question=request.follow_up_question,
                document=explanation,
                tool_results=request.tool_results,
                history=history,
                language=request.language,
                explanation_style=request.explanation_style,
            )
        )
        return DemoFlowResult(
            ocr=ocr,
            explanation=explanation,
            follow_up=follow_up,
        )
