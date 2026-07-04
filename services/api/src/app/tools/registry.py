from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from app.tools.base import CitizenTool, ToolDefinition, ValidatedToolCall
from app.tools.calculator import AddAmountsTool


class ToolValidationError(ValueError):
    def __init__(self, code: str, message: str, *, tool_name: str) -> None:
        super().__init__(message)
        self.code = code
        self.tool_name = tool_name


class ToolRegistry:
    def __init__(self, tools: Iterable[CitizenTool]) -> None:
        self._tools: dict[str, CitizenTool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def definitions(self) -> list[ToolDefinition]:
        return [self._tools[name].definition() for name in sorted(self._tools)]

    def validate_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ValidatedToolCall:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolValidationError(
                "tool_not_allowed",
                f"Tool '{tool_name}' is not allowed.",
                tool_name=tool_name,
            )

        try:
            validated = tool.argument_model.model_validate(arguments)
        except ValidationError as error:
            raise ToolValidationError(
                "invalid_tool_arguments",
                f"Arguments for tool '{tool_name}' are invalid.",
                tool_name=tool_name,
            ) from error

        return ValidatedToolCall(
            tool_name=tool.name,
            arguments=validated.model_dump(mode="json"),
            requires_confirmation=tool.requires_confirmation,
            creates_external_side_effect=tool.creates_external_side_effect,
        )

    def get(self, tool_name: str) -> CitizenTool:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolValidationError(
                "tool_not_allowed",
                f"Tool '{tool_name}' is not allowed.",
                tool_name=tool_name,
            )
        return tool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry([AddAmountsTool()])
