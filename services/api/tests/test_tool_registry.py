from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from app.tools.base import CitizenTool, ToolResult
from app.tools.registry import ToolRegistry, ToolValidationError, build_tool_registry


def test_registry_exposes_only_allow_listed_strict_schemas() -> None:
    definitions = build_tool_registry().definitions()

    assert [definition.name for definition in definitions] == ["add_amounts"]
    definition = definitions[0]
    assert definition.requires_confirmation is False
    assert definition.creates_external_side_effect is False
    assert definition.input_schema["required"] == ["amounts"]
    assert definition.input_schema["additionalProperties"] is False
    assert definition.input_schema["properties"]["amounts"]["minItems"] == 1
    assert definition.input_schema["properties"]["amounts"]["maxItems"] == 100


def test_valid_arguments_are_normalized_before_execution() -> None:
    call = build_tool_registry().validate_call(
        tool_name="add_amounts",
        arguments={"amounts": ["10.50", 2, Decimal("0.25")]},
    )

    assert call.tool_name == "add_amounts"
    assert call.arguments == {"amounts": ["10.50", "2", "0.25"]}
    assert call.requires_confirmation is False
    assert call.creates_external_side_effect is False


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"amounts": []},
        {"amounts": ["not-a-number"]},
        {"amounts": ["NaN"]},
        {"amounts": ["Infinity"]},
        {"amounts": [1], "unexpected": True},
    ],
)
def test_invalid_arguments_are_rejected_before_execution(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ToolValidationError) as error:
        build_tool_registry().validate_call(
            tool_name="add_amounts",
            arguments=arguments,
        )

    assert error.value.code == "invalid_tool_arguments"
    assert error.value.tool_name == "add_amounts"


def test_unknown_tool_is_rejected_by_allow_list() -> None:
    with pytest.raises(ToolValidationError) as error:
        build_tool_registry().validate_call(
            tool_name="run_arbitrary_code",
            arguments={"code": "dangerous()"},
        )

    assert error.value.code == "tool_not_allowed"
    assert error.value.tool_name == "run_arbitrary_code"


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DuplicateTool(CitizenTool):
    name = "duplicate"
    description = "A duplicate-name test tool."
    argument_model = EmptyArguments

    async def execute(self, arguments: BaseModel) -> ToolResult:
        return ToolResult(ok=True, summary="done")


def test_duplicate_tool_names_are_rejected_at_startup() -> None:
    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolRegistry([DuplicateTool(), DuplicateTool()])
