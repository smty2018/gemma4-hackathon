from collections.abc import Sequence
from typing import Any

import pytest

from app.inference.gemma import (
    DEFAULT_MODEL_ID,
    GemmaInferenceError,
    GemmaRequest,
    GemmaResponse,
    GemmaToolCall,
)
from app.tools.executor import ProposalStatus, ToolExecutor
from app.tools.planner import (
    MAX_TOOL_CONTEXT_CHARACTERS,
    MAX_TOOL_REQUEST_CHARACTERS,
    ToolPlanner,
    ToolPlanningError,
    ToolPlanningRequest,
)
from app.tools.registry import build_tool_registry


class StubGemma:
    def __init__(self, responses: Sequence[GemmaResponse | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[GemmaRequest] = []

    def generate(self, request: GemmaRequest) -> GemmaResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def tool_response(
    *,
    name: str | None = None,
    arguments: dict[str, Any] | None = None,
    text: str = "",
) -> GemmaResponse:
    calls = ()
    if name is not None:
        calls = (GemmaToolCall(name=name, arguments=arguments or {}),)
    return GemmaResponse(
        model_id=DEFAULT_MODEL_ID,
        text=text,
        tool_calls=calls,
    )


def planner_for(gemma: StubGemma) -> ToolPlanner:
    registry = build_tool_registry()
    return ToolPlanner(
        gemma=gemma,
        registry=registry,
        executor=ToolExecutor(registry),
    )


def test_gemma_selection_becomes_a_validated_proposal_not_execution() -> None:
    gemma = StubGemma(
        [
            tool_response(
                name="add_amounts",
                arguments={"amounts": ["10.50", 2]},
                text="Add the two stated amounts.",
            )
        ]
    )

    decision = planner_for(gemma).plan(
        ToolPlanningRequest(
            actor_id="session-1",
            user_request="What is 10.50 plus 2?",
            context="The document lists 10.50 and 2.",
        )
    )

    assert decision.reason == "Add the two stated amounts."
    assert decision.proposal is not None
    assert decision.proposal.tool_name == "add_amounts"
    assert decision.proposal.arguments == {"amounts": ["10.50", "2"]}
    assert decision.proposal.status is ProposalStatus.READY

    request = gemma.requests[0]
    assert request.temperature == 0
    assert request.max_new_tokens == 512
    assert request.response_schema is None
    assert request.enable_thinking is True
    assert request.system_instruction is not None
    assert request.tools[0]["function"]["name"] == "add_amounts"
    assert request.tools[0]["function"]["parameters"]["additionalProperties"] is False
    assert "What is 10.50 plus 2?" in request.prompt


def test_model_can_decide_no_tool_is_needed() -> None:
    gemma = StubGemma(
        [tool_response(text="This only needs a plain-language answer.")]
    )

    decision = planner_for(gemma).plan(
        ToolPlanningRequest(actor_id="session-1", user_request="Explain this notice.")
    )

    assert decision.proposal is None
    assert "plain-language" in decision.reason


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("run_arbitrary_code", {"code": "dangerous()"}),
        ("add_amounts", {"amounts": []}),
        ("add_amounts", {"amounts": [1], "unexpected": True}),
    ],
)
def test_hallucinated_tools_and_invalid_arguments_never_become_proposals(
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    gemma = StubGemma(
        [tool_response(name=tool_name, arguments=arguments, text="Attempt a tool call.")]
    )

    with pytest.raises(ToolPlanningError) as error:
        planner_for(gemma).plan(
            ToolPlanningRequest(actor_id="session-1", user_request="Do this.")
        )

    assert error.value.code == "invalid_tool_selection"
    assert error.value.retryable is True


def test_multiple_native_tool_calls_are_rejected() -> None:
    response = GemmaResponse(
        model_id=DEFAULT_MODEL_ID,
        tool_calls=(
            GemmaToolCall(name="add_amounts", arguments={"amounts": [1]}),
            GemmaToolCall(name="add_amounts", arguments={"amounts": [2]}),
        ),
    )
    with pytest.raises(ToolPlanningError) as error:
        planner_for(StubGemma([response])).plan(
            ToolPlanningRequest(actor_id="session-1", user_request="Do this.")
        )

    assert error.value.code == "tool_planning_response_invalid"


def test_gemma_errors_are_sanitized() -> None:
    gemma = StubGemma(
        [
            GemmaInferenceError(
                "gemma_inference_failed",
                "private GPU details",
                retryable=True,
            )
        ]
    )

    with pytest.raises(ToolPlanningError) as error:
        planner_for(gemma).plan(
            ToolPlanningRequest(actor_id="session-1", user_request="Add 1 and 2.")
        )

    assert error.value.code == "tool_planning_inference_failed"
    assert error.value.retryable is True
    assert "private GPU details" not in str(error.value)


@pytest.mark.parametrize(
    ("planning_request", "code"),
    [
        (
            ToolPlanningRequest(actor_id="", user_request="Add 1 and 2."),
            "invalid_tool_actor",
        ),
        (
            ToolPlanningRequest(actor_id="session-1", user_request=""),
            "invalid_tool_request",
        ),
        (
            ToolPlanningRequest(
                actor_id="session-1",
                user_request="x" * (MAX_TOOL_REQUEST_CHARACTERS + 1),
            ),
            "invalid_tool_request",
        ),
        (
            ToolPlanningRequest(
                actor_id="session-1",
                user_request="Add values.",
                context="x" * (MAX_TOOL_CONTEXT_CHARACTERS + 1),
            ),
            "tool_context_too_large",
        ),
        (
            ToolPlanningRequest(
                actor_id="session-1",
                user_request="Add values.",
                language="x",
            ),
            "invalid_tool_language",
        ),
    ],
)
def test_invalid_planning_inputs_stop_before_gemma(
    planning_request: ToolPlanningRequest,
    code: str,
) -> None:
    gemma = StubGemma([])

    with pytest.raises(ToolPlanningError) as error:
        planner_for(gemma).plan(planning_request)

    assert error.value.code == code
    assert gemma.requests == []
