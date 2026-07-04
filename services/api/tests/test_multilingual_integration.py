import asyncio
from dataclasses import dataclass

import pytest

from app.explanation.pipeline import ExplanationPipeline, ExplanationRequest
from app.inference.gemma import (
    DEFAULT_MODEL_ID,
    GemmaRequest,
    GemmaResponse,
    GemmaToolCall,
)
from app.ocr.pipeline import OcrPageInput, OcrPipeline
from app.tools.executor import ToolExecutor
from app.tools.planner import ToolPlanner, ToolPlanningRequest
from app.tools.registry import build_tool_registry


@dataclass(frozen=True)
class LanguageScenario:
    language: str
    source_text: str
    document_label: str
    date_label: str
    amount_label: str
    summary: str
    fact_label: str
    action: str
    script_fragment: str


SCENARIOS = (
    LanguageScenario(
        language="English",
        source_text="Payment notice\nDue date: 10 July 2026\nAmount due: ₹1,250",
        document_label="Payment notice",
        date_label="Due date: 10 July 2026",
        amount_label="Amount due: ₹1,250",
        summary="This payment notice states an amount and a due date.",
        fact_label="Amount due",
        action="Pay ₹1,250 by 10 July 2026.",
        script_fragment="payment",
    ),
    LanguageScenario(
        language="Hindi",
        source_text="भुगतान सूचना\nअंतिम तिथि: 10 जुलाई 2026\nदेय राशि: ₹1,250",
        document_label="भुगतान सूचना",
        date_label="अंतिम तिथि: 10 जुलाई 2026",
        amount_label="देय राशि: ₹1,250",
        summary="यह भुगतान सूचना राशि और अंतिम तिथि बताती है।",
        fact_label="देय राशि",
        action="10 जुलाई 2026 तक ₹1,250 का भुगतान करें।",
        script_fragment="भुगतान",
    ),
    LanguageScenario(
        language="Bengali",
        source_text="পেমেন্ট নোটিশ\nশেষ তারিখ: 10 জুলাই 2026\nবকেয়া পরিমাণ: ₹1,250",
        document_label="পেমেন্ট নোটিশ",
        date_label="শেষ তারিখ: 10 জুলাই 2026",
        amount_label="বকেয়া পরিমাণ: ₹1,250",
        summary="এই পেমেন্ট নোটিশে পরিমাণ ও শেষ তারিখ বলা আছে।",
        fact_label="বকেয়া পরিমাণ",
        action="10 জুলাই 2026-এর মধ্যে ₹1,250 পরিশোধ করুন।",
        script_fragment="পেমেন্ট",
    ),
)


class MultilingualGemma4Stub:
    def __init__(self, scenario: LanguageScenario) -> None:
        self.scenario = scenario
        self.requests: list[GemmaRequest] = []

    def generate(self, request: GemmaRequest) -> GemmaResponse:
        self.requests.append(request)
        schema_name = getattr(request.response_schema, "__name__", "")

        if schema_name == "_PageExtractionPayload":
            return GemmaResponse(
                model_id=DEFAULT_MODEL_ID,
                structured={
                    "text": self.scenario.source_text,
                    "evidence": [
                        {
                            "label": "Document type",
                            "value": self.scenario.document_label,
                            "evidence_text": self.scenario.document_label,
                            "confidence": 0.95,
                        }
                    ],
                    "dates": [
                        {
                            "value": "10 July 2026",
                            "normalized_value": "2026-07-10",
                            "evidence_text": self.scenario.date_label,
                            "confidence": 0.93,
                        }
                    ],
                    "amounts": [
                        {
                            "value": "₹1,250",
                            "normalized_value": "1250",
                            "currency": "INR",
                            "evidence_text": self.scenario.amount_label,
                            "confidence": 0.96,
                        }
                    ],
                    "confidence": 0.94,
                },
            )

        if schema_name == "_ExplanationPayload":
            return GemmaResponse(
                model_id=DEFAULT_MODEL_ID,
                structured={
                    "simple_summary": {
                        "text": self.scenario.summary,
                        "evidence_ids": ["P1"],
                    },
                    "key_facts": [
                        {
                            "label": self.scenario.fact_label,
                            "value": "₹1,250",
                            "evidence_ids": ["E3"],
                            "confidence": 0.95,
                        }
                    ],
                    "required_actions": [
                        {
                            "action": self.scenario.action,
                            "deadline": "10 July 2026",
                            "urgency": "soon",
                            "evidence_ids": ["E2", "E3"],
                            "confidence": 0.92,
                        }
                    ],
                    "warnings": [],
                    "confidence": 0.93,
                },
            )

        if request.tools:
            return GemmaResponse(
                model_id=DEFAULT_MODEL_ID,
                text=f"{self.scenario.language}: calculate the combined amount.",
                tool_calls=(
                    GemmaToolCall(
                        name="add_amounts",
                        arguments={"amounts": ["1250", "250"]},
                    ),
                ),
            )

        raise AssertionError("Unexpected integration request")


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.language)
def test_document_flow_in_english_hindi_and_bengali(
    scenario: LanguageScenario,
) -> None:
    gemma = MultilingualGemma4Stub(scenario)
    ocr = OcrPipeline(gemma).extract(
        [OcrPageInput(page=1, embedded_text=scenario.source_text)]
    )
    explanation = ExplanationPipeline(gemma).explain(
        ExplanationRequest(
            ocr=ocr,
            language=scenario.language,
            audience="general public",
        )
    )

    registry = build_tool_registry()
    executor = ToolExecutor(registry)
    decision = ToolPlanner(
        gemma=gemma,
        registry=registry,
        executor=executor,
    ).plan(
        ToolPlanningRequest(
            actor_id=f"integration-{scenario.language.lower()}",
            user_request="Add 1,250 and 250.",
            context=explanation.simple_summary.text,
            language=scenario.language,
        )
    )
    assert decision.proposal is not None
    receipt = asyncio.run(
        executor.execute(
            actor_id=f"integration-{scenario.language.lower()}",
            proposal_id=decision.proposal.proposal_id,
        )
    )

    assert DEFAULT_MODEL_ID == "google/gemma-4-E4B-it"
    assert ocr.amounts[0].normalized_value == "1250"
    assert ocr.dates[0].normalized_value == "2026-07-10"
    assert ocr.amounts[0].grounded is True
    assert scenario.script_fragment.casefold() in explanation.simple_summary.text.casefold()
    assert explanation.language == scenario.language
    assert explanation.required_actions[0].evidence_ids == ["E2", "E3"]
    assert {item.evidence_id for item in explanation.source_evidence} >= {
        "P1",
        "E1",
        "E2",
        "E3",
    }
    assert receipt.result.data == {"total": "1500"}
    assert len(gemma.requests) == 3
    assert gemma.requests[2].tools[0]["function"]["name"] == "add_amounts"
    assert gemma.requests[2].enable_thinking is True
