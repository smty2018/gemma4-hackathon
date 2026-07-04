import copy
import hashlib
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import ToolResult, ValidatedToolCall
from app.tools.registry import ToolRegistry

DEFAULT_PROPOSAL_TTL_SECONDS = 300
MIN_PROPOSAL_TTL_SECONDS = 10
MAX_PROPOSAL_TTL_SECONDS = 3_600


class ProposalStatus(StrEnum):
    PENDING_CONFIRMATION = "pending_confirmation"
    READY = "ready"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


class ToolProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str = Field(min_length=16, max_length=200)
    tool_name: str
    arguments: dict[str, Any]
    requires_confirmation: bool
    creates_external_side_effect: bool
    status: ProposalStatus
    expires_at: datetime


class ToolExecutionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult
    executed_at: datetime


class ToolExecutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        proposal_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.proposal_id = proposal_id
        self.retryable = retryable


@dataclass
class _StoredProposal:
    call: ValidatedToolCall
    actor_digest: bytes
    expires_at: datetime
    status: ProposalStatus


Clock = Callable[[], datetime]
TokenFactory = Callable[[], str]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        proposal_ttl_seconds: int = DEFAULT_PROPOSAL_TTL_SECONDS,
        clock: Clock | None = None,
        token_factory: TokenFactory | None = None,
    ) -> None:
        if not MIN_PROPOSAL_TTL_SECONDS <= proposal_ttl_seconds <= MAX_PROPOSAL_TTL_SECONDS:
            raise ValueError(
                f"proposal_ttl_seconds must be between {MIN_PROPOSAL_TTL_SECONDS} "
                f"and {MAX_PROPOSAL_TTL_SECONDS}"
            )
        self._registry = registry
        self._proposal_ttl = timedelta(seconds=proposal_ttl_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(24))
        self._proposals: dict[str, _StoredProposal] = {}
        self._lock = Lock()

    def prepare(
        self,
        *,
        actor_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolProposal:
        actor_digest = _actor_digest(actor_id)
        call = self._registry.validate_call(
            tool_name=tool_name,
            arguments=arguments,
        )
        now = self._current_time()
        expires_at = now + self._proposal_ttl
        status = (
            ProposalStatus.PENDING_CONFIRMATION
            if call.requires_confirmation
            else ProposalStatus.READY
        )

        with self._lock:
            proposal_id = self._unique_proposal_id()
            self._proposals[proposal_id] = _StoredProposal(
                call=ValidatedToolCall(
                    tool_name=call.tool_name,
                    arguments=copy.deepcopy(call.arguments),
                    requires_confirmation=call.requires_confirmation,
                    creates_external_side_effect=call.creates_external_side_effect,
                ),
                actor_digest=actor_digest,
                expires_at=expires_at,
                status=status,
            )
            return self._proposal_view(proposal_id, self._proposals[proposal_id])

    def confirm(self, *, actor_id: str, proposal_id: str) -> ToolProposal:
        actor_digest = _actor_digest(actor_id)
        with self._lock:
            stored = self._authorized_proposal(proposal_id, actor_digest)
            self._ensure_not_expired(proposal_id, stored)
            if not stored.call.requires_confirmation:
                raise ToolExecutionError(
                    "confirmation_not_required",
                    "This tool call does not require confirmation.",
                    proposal_id=proposal_id,
                )
            if stored.status == ProposalStatus.PENDING_CONFIRMATION:
                stored.status = ProposalStatus.READY
                return self._proposal_view(proposal_id, stored)
            if stored.status == ProposalStatus.READY:
                raise ToolExecutionError(
                    "proposal_already_confirmed",
                    "This tool call has already been confirmed.",
                    proposal_id=proposal_id,
                )
            raise self._state_error(proposal_id, stored.status)

    async def execute(
        self,
        *,
        actor_id: str,
        proposal_id: str,
    ) -> ToolExecutionReceipt:
        actor_digest = _actor_digest(actor_id)
        with self._lock:
            stored = self._authorized_proposal(proposal_id, actor_digest)
            self._ensure_not_expired(proposal_id, stored)
            if stored.status == ProposalStatus.PENDING_CONFIRMATION:
                raise ToolExecutionError(
                    "tool_confirmation_required",
                    "Confirm this exact tool call before execution.",
                    proposal_id=proposal_id,
                )
            if stored.status != ProposalStatus.READY:
                raise self._state_error(proposal_id, stored.status)
            stored.status = ProposalStatus.EXECUTING
            call = ValidatedToolCall(
                tool_name=stored.call.tool_name,
                arguments=copy.deepcopy(stored.call.arguments),
                requires_confirmation=stored.call.requires_confirmation,
                creates_external_side_effect=stored.call.creates_external_side_effect,
            )

        try:
            tool = self._registry.get(call.tool_name)
            validated_arguments = tool.argument_model.model_validate(call.arguments)
            result = ToolResult.model_validate(await tool.execute(validated_arguments))
        except Exception as error:
            with self._lock:
                stored.status = ProposalStatus.FAILED
            raise ToolExecutionError(
                "tool_execution_failed",
                f"Tool '{call.tool_name}' could not be executed.",
                proposal_id=proposal_id,
            ) from error

        executed_at = self._current_time()
        with self._lock:
            stored.status = ProposalStatus.EXECUTED
        return ToolExecutionReceipt(
            proposal_id=proposal_id,
            tool_name=call.tool_name,
            arguments=copy.deepcopy(call.arguments),
            result=result,
            executed_at=executed_at,
        )

    def _unique_proposal_id(self) -> str:
        for _ in range(10):
            proposal_id = self._token_factory()
            if 16 <= len(proposal_id) <= 200 and proposal_id not in self._proposals:
                return proposal_id
        raise ToolExecutionError(
            "proposal_id_generation_failed",
            "A unique tool proposal could not be created.",
            retryable=True,
        )

    def _authorized_proposal(
        self,
        proposal_id: str,
        actor_digest: bytes,
    ) -> _StoredProposal:
        stored = self._proposals.get(proposal_id)
        if stored is None or not secrets.compare_digest(stored.actor_digest, actor_digest):
            raise ToolExecutionError(
                "tool_proposal_not_found",
                "The tool proposal was not found for this session.",
                proposal_id=proposal_id,
            )
        return stored

    def _ensure_not_expired(
        self,
        proposal_id: str,
        stored: _StoredProposal,
    ) -> None:
        if stored.status in {ProposalStatus.EXECUTED, ProposalStatus.FAILED}:
            return
        if self._current_time() >= stored.expires_at:
            stored.status = ProposalStatus.EXPIRED
            raise ToolExecutionError(
                "tool_proposal_expired",
                "The tool proposal expired and must be prepared again.",
                proposal_id=proposal_id,
            )

    @staticmethod
    def _proposal_view(
        proposal_id: str,
        stored: _StoredProposal,
    ) -> ToolProposal:
        return ToolProposal(
            proposal_id=proposal_id,
            tool_name=stored.call.tool_name,
            arguments=copy.deepcopy(stored.call.arguments),
            requires_confirmation=stored.call.requires_confirmation,
            creates_external_side_effect=stored.call.creates_external_side_effect,
            status=stored.status,
            expires_at=stored.expires_at,
        )

    @staticmethod
    def _state_error(
        proposal_id: str,
        status: ProposalStatus,
    ) -> ToolExecutionError:
        codes = {
            ProposalStatus.EXECUTING: (
                "tool_execution_in_progress",
                "This tool call is already executing.",
            ),
            ProposalStatus.EXECUTED: (
                "tool_proposal_already_executed",
                "This tool call has already been executed.",
            ),
            ProposalStatus.FAILED: (
                "tool_proposal_failed",
                "This tool call already failed and cannot be replayed.",
            ),
            ProposalStatus.EXPIRED: (
                "tool_proposal_expired",
                "This tool proposal has expired.",
            ),
        }
        code, message = codes.get(
            status,
            ("invalid_tool_proposal_state", "This tool proposal cannot be used."),
        )
        return ToolExecutionError(code, message, proposal_id=proposal_id)

    def _current_time(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Tool executor clock must return a timezone-aware datetime")
        return now


def _actor_digest(actor_id: str) -> bytes:
    normalized = actor_id.strip()
    if not normalized or len(normalized) > 200:
        raise ToolExecutionError(
            "invalid_tool_actor",
            "A valid session or actor identifier is required.",
        )
    return hashlib.sha256(normalized.encode("utf-8")).digest()
