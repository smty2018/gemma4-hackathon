from decimal import Decimal, InvalidOperation
from typing import Any

from app.tools.base import CitizenTool, ToolResult


class AddAmountsTool(CitizenTool):
    name = "add_amounts"

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            amounts = [Decimal(str(value)) for value in arguments.get("amounts", [])]
        except InvalidOperation:
            return ToolResult(ok=False, summary="One or more amounts are invalid.")

        total = sum(amounts, start=Decimal("0"))
        return ToolResult(ok=True, summary=f"Total: {total}", data={"total": str(total)})
