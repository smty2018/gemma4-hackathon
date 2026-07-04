from collections.abc import Sequence
from typing import Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.tools.base import CitizenTool, ToolResult

DEFAULT_OFFICIAL_DOMAIN_SUFFIXES = (".gov.in", ".nic.in")


class OfficialSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=500)
    jurisdiction: str = Field(min_length=2, max_length=100)
    language: str = Field(default="English", min_length=2, max_length=32)
    max_results: int = Field(default=5, ge=1, le=10)


class OfficialSearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=300)
    url: HttpUrl
    authority: str = Field(min_length=1, max_length=200)
    snippet: str = Field(min_length=1, max_length=1_000)


class OfficialSearchProvider(Protocol):
    async def search(
        self,
        *,
        query: str,
        jurisdiction: str,
        language: str,
        limit: int,
    ) -> list[OfficialSearchHit]: ...


class OfficialSearchTool(CitizenTool):
    name = "search_official_sources"
    description = (
        "Search configured official government sources and return verified HTTPS results."
    )
    argument_model = OfficialSearchArguments
    requires_confirmation = False

    def __init__(
        self,
        provider: OfficialSearchProvider,
        *,
        allowed_domain_suffixes: Sequence[str] = DEFAULT_OFFICIAL_DOMAIN_SUFFIXES,
    ) -> None:
        suffixes = tuple(_normalize_domain_suffix(value) for value in allowed_domain_suffixes)
        if not suffixes:
            raise ValueError("At least one official domain suffix is required")
        self._provider = provider
        self._allowed_domain_suffixes = suffixes

    async def execute(self, arguments: BaseModel) -> ToolResult:
        validated = OfficialSearchArguments.model_validate(arguments)
        provider_hits = await self._provider.search(
            query=validated.query,
            jurisdiction=validated.jurisdiction,
            language=validated.language,
            limit=validated.max_results,
        )
        verified_hits = [
            hit
            for hit in provider_hits
            if _is_allowed_official_url(
                str(hit.url),
                self._allowed_domain_suffixes,
            )
        ][: validated.max_results]

        if not verified_hits:
            return ToolResult(
                ok=False,
                summary="No verified official results were found.",
                data={"results": []},
            )

        return ToolResult(
            ok=True,
            summary=f"Found {len(verified_hits)} verified official result(s).",
            data={
                "results": [hit.model_dump(mode="json") for hit in verified_hits],
            },
        )


def _normalize_domain_suffix(value: str) -> str:
    normalized = value.strip().lower().rstrip(".")
    if not normalized:
        raise ValueError("Official domain suffixes cannot be empty")
    return "." + normalized.lstrip(".")


def _is_allowed_official_url(url: str, allowed_suffixes: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower().rstrip(".")
    return any(
        hostname == suffix.lstrip(".") or hostname.endswith(suffix)
        for suffix in allowed_suffixes
    )
