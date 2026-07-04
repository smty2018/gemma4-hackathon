from app.tools.base import CitizenTool
from app.tools.calculator import AddAmountsTool


def build_tool_registry() -> dict[str, CitizenTool]:
    tools: list[CitizenTool] = [AddAmountsTool()]
    return {tool.name: tool for tool in tools}
