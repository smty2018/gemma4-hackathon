from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import ValidationError

from app.tools.base import CitizenTool, ToolDefinition, ValidatedToolCall
from app.tools.calculator import AddAmountsTool
from app.tools.official_search import OfficialSearchProvider, OfficialSearchTool


class ToolValidationError(ValueError):
    def __init__(self, code: str, message: str, *, tool_name: str) -> None:
        super().__init__(message)
        self.code = code
        self.tool_name = tool_name


class ToolRegistry:
    def __init__(self, tools: Iterable[CitizenTool]) -> None:
        self._tools: dict[str, CitizenTool] = {}
        for tool in tools:
            if tool.creates_external_side_effect and not tool.requires_confirmation:
                raise ValueError(
                    f"Side-effecting tool '{tool.name}' must require confirmation"
                )
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

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


def build_tool_registry(
    *,
    official_search_provider: OfficialSearchProvider | None = None,
    official_domain_suffixes: Sequence[str] = (".gov.in", ".nic.in"),
) -> ToolRegistry:
    tools: list[CitizenTool] = [AddAmountsTool()]
    if official_search_provider is not None:
        tools.append(
            OfficialSearchTool(
                official_search_provider,
                allowed_domain_suffixes=official_domain_suffixes,
            )
        )
    return ToolRegistry(tools)
