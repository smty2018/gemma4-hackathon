from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class CitizenTool(ABC):
    name: str
    creates_external_side_effect: bool = False

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate arguments and execute the tool."""
