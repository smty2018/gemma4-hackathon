import asyncio
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import CitizenTool, ToolResult
from app.tools.executor import ProposalStatus, ToolExecutionError, ToolExecutor
from app.tools.registry import ToolRegistry, build_tool_registry


class SendNoticeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=500)


class SendNoticeTool(CitizenTool):
    name = "send_notice"
    description = "Send a notice to an external destination."
    argument_model = SendNoticeArguments
    creates_external_side_effect = True
    requires_confirmation = True
    calls: ClassVar[list[SendNoticeArguments]] = []
    failure: ClassVar[Exception | None] = None

    async def execute(self, arguments: BaseModel) -> ToolResult:
        validated = SendNoticeArguments.model_validate(arguments)
        self.calls.append(validated)
        if self.failure:
            raise self.failure
        return ToolResult(
            ok=True,
            summary=f"Notice sent to {validated.destination}.",
            data={"destination": validated.destination},
        )


class UnsafeTool(SendNoticeTool):
    name = "unsafe_notice"
    requires_confirmation = False


@pytest.fixture(autouse=True)
def clear_fake_tool_state() -> None:
    SendNoticeTool.calls = []
    SendNoticeTool.failure = None


def executor_for_side_effects(**kwargs) -> ToolExecutor:
    return ToolExecutor(ToolRegistry([SendNoticeTool()]), **kwargs)


def test_side_effecting_tools_must_require_confirmation() -> None:
    with pytest.raises(ValueError, match="must require confirmation"):
        ToolRegistry([UnsafeTool()])


def test_prepare_shows_exact_normalized_call_without_executing() -> None:
    executor = executor_for_side_effects(token_factory=lambda: "proposal-token-123456")

    proposal = executor.prepare(
        actor_id="session-1",
        tool_name="send_notice",
        arguments={"destination": "office", "message": "Please review"},
    )

    assert proposal.proposal_id == "proposal-token-123456"
    assert proposal.status == ProposalStatus.PENDING_CONFIRMATION
    assert proposal.requires_confirmation is True
    assert proposal.creates_external_side_effect is True
    assert proposal.arguments == {
        "destination": "office",
        "message": "Please review",
    }
    assert SendNoticeTool.calls == []


def test_execution_is_blocked_until_exact_proposal_is_confirmed() -> None:
    executor = executor_for_side_effects()
    proposal = executor.prepare(
        actor_id="session-1",
        tool_name="send_notice",
        arguments={"destination": "office", "message": "Please review"},
    )

    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(
            executor.execute(
                actor_id="session-1",
                proposal_id=proposal.proposal_id,
            )
        )

    assert error.value.code == "tool_confirmation_required"
    assert SendNoticeTool.calls == []


def test_confirmation_and_execution_are_actor_bound_and_single_use() -> None:
    executor = executor_for_side_effects()
    proposal = executor.prepare(
        actor_id="session-1",
        tool_name="send_notice",
        arguments={"destination": "office", "message": "Please review"},
    )

    with pytest.raises(ToolExecutionError) as wrong_actor:
        executor.confirm(actor_id="session-2", proposal_id=proposal.proposal_id)
    assert wrong_actor.value.code == "tool_proposal_not_found"

    confirmed = executor.confirm(
        actor_id="session-1",
        proposal_id=proposal.proposal_id,
    )
    assert confirmed.status == ProposalStatus.READY

    receipt = asyncio.run(
        executor.execute(actor_id="session-1", proposal_id=proposal.proposal_id)
    )
    assert receipt.result.ok is True
    assert receipt.result.summary == "Notice sent to office."
    assert receipt.arguments == {
        "destination": "office",
        "message": "Please review",
    }
    assert len(SendNoticeTool.calls) == 1

    with pytest.raises(ToolExecutionError) as replay:
        asyncio.run(
            executor.execute(actor_id="session-1", proposal_id=proposal.proposal_id)
        )
    assert replay.value.code == "tool_proposal_already_executed"
    assert len(SendNoticeTool.calls) == 1


def test_mutating_returned_proposal_does_not_change_stored_arguments() -> None:
    executor = executor_for_side_effects()
    proposal = executor.prepare(
        actor_id="session-1",
        tool_name="send_notice",
        arguments={"destination": "office", "message": "Original"},
    )
    proposal.arguments["destination"] = "attacker"
    executor.confirm(actor_id="session-1", proposal_id=proposal.proposal_id)

    receipt = asyncio.run(
        executor.execute(actor_id="session-1", proposal_id=proposal.proposal_id)
    )

    assert receipt.arguments["destination"] == "office"
    assert SendNoticeTool.calls[0].destination == "office"


def test_expired_proposal_cannot_be_confirmed_or_executed() -> None:
    now = datetime(2026, 7, 5, tzinfo=UTC)
    clock_value = [now]
    executor = executor_for_side_effects(
        proposal_ttl_seconds=10,
        clock=lambda: clock_value[0],
    )
    proposal = executor.prepare(
        actor_id="session-1",
        tool_name="send_notice",
        arguments={"destination": "office", "message": "Original"},
    )
    clock_value[0] = now + timedelta(seconds=10)

    with pytest.raises(ToolExecutionError) as error:
        executor.confirm(actor_id="session-1", proposal_id=proposal.proposal_id)

    assert error.value.code == "tool_proposal_expired"
    assert SendNoticeTool.calls == []


def test_non_side_effecting_tool_can_execute_without_confirmation() -> None:
    executor = ToolExecutor(build_tool_registry())
    proposal = executor.prepare(
        actor_id="session-1",
        tool_name="add_amounts",
        arguments={"amounts": ["10.50", "2.25"]},
    )

    assert proposal.status == ProposalStatus.READY
    receipt = asyncio.run(
        executor.execute(actor_id="session-1", proposal_id=proposal.proposal_id)
    )

    assert receipt.result.ok is True
    assert receipt.result.data == {"total": "12.75"}


def test_tool_failures_are_normalized_and_not_replayable() -> None:
    SendNoticeTool.failure = RuntimeError("private provider details")
    executor = executor_for_side_effects()
    proposal = executor.prepare(
        actor_id="session-1",
        tool_name="send_notice",
        arguments={"destination": "office", "message": "Original"},
    )
    executor.confirm(actor_id="session-1", proposal_id=proposal.proposal_id)

    with pytest.raises(ToolExecutionError) as first_error:
        asyncio.run(
            executor.execute(actor_id="session-1", proposal_id=proposal.proposal_id)
        )
    assert first_error.value.code == "tool_execution_failed"
    assert "private provider details" not in str(first_error.value)

    with pytest.raises(ToolExecutionError) as replay_error:
        asyncio.run(
            executor.execute(actor_id="session-1", proposal_id=proposal.proposal_id)
        )
    assert replay_error.value.code == "tool_proposal_failed"


@pytest.mark.parametrize("actor_id", ["", " ", "x" * 201])
def test_invalid_actor_context_is_rejected(actor_id: str) -> None:
    with pytest.raises(ToolExecutionError) as error:
        executor_for_side_effects().prepare(
            actor_id=actor_id,
            tool_name="send_notice",
            arguments={"destination": "office", "message": "Original"},
        )

    assert error.value.code == "invalid_tool_actor"
