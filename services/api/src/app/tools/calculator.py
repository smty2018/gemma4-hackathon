from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.tools.base import CitizenTool, ToolResult


class AddAmountsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amounts: list[Decimal] = Field(min_length=1, max_length=100)

    @field_validator("amounts")
    @classmethod
    def amounts_must_be_finite(cls, values: list[Decimal]) -> list[Decimal]:
        if any(not value.is_finite() for value in values):
            raise ValueError("amounts must be finite")
        return values


class AddAmountsTool(CitizenTool):
    name = "add_amounts"
    description = "Add a list of monetary or decimal amounts without changing external state."
    argument_model = AddAmountsArguments
    requires_confirmation = False

    async def execute(self, arguments: BaseModel) -> ToolResult:
        validated = AddAmountsArguments.model_validate(arguments)
        total = sum(validated.amounts, start=Decimal("0"))
        return ToolResult(ok=True, summary=f"Total: {total}", data={"total": str(total)})
