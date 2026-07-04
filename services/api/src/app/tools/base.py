from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    summary: str = Field(min_length=1, max_length=1_000)
    data: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    description: str = Field(min_length=1, max_length=500)
    input_schema: dict[str, Any]
    creates_external_side_effect: bool
    requires_confirmation: bool


@dataclass(frozen=True)
class ValidatedToolCall:
    tool_name: str
    arguments: dict[str, Any]
    requires_confirmation: bool
    creates_external_side_effect: bool


class CitizenTool(ABC):
    name: str
    description: str
    argument_model: type[BaseModel]
    creates_external_side_effect: bool = False
    requires_confirmation: bool = True

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.argument_model.model_json_schema(),
            creates_external_side_effect=self.creates_external_side_effect,
            requires_confirmation=self.requires_confirmation,
        )

    @abstractmethod
    async def execute(self, arguments: BaseModel) -> ToolResult:
        """Execute already validated arguments."""
