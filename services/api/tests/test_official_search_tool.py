import asyncio
from typing import Any

import pytest

from app.tools.executor import ToolExecutor
from app.tools.official_search import OfficialSearchHit, OfficialSearchTool
from app.tools.registry import ToolValidationError, build_tool_registry


class StubOfficialSearchProvider:
    def __init__(self, hits: list[OfficialSearchHit]) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []

    async def search(
        self,
        *,
        query: str,
        jurisdiction: str,
        language: str,
        limit: int,
    ) -> list[OfficialSearchHit]:
        self.calls.append(
            {
                "query": query,
                "jurisdiction": jurisdiction,
                "language": language,
                "limit": limit,
            }
        )
        return self.hits


def search_arguments() -> dict[str, Any]:
    return {
        "query": "income certificate application",
        "jurisdiction": "Karnataka, India",
        "language": "English",
        "max_results": 2,
    }


def test_registry_keeps_calculator_first_and_adds_configured_search() -> None:
    provider = StubOfficialSearchProvider([])
    definitions = build_tool_registry(
        official_search_provider=provider,
    ).definitions()

    assert [definition.name for definition in definitions] == [
        "add_amounts",
        "search_official_sources",
    ]
    search_schema = definitions[1].input_schema
    assert search_schema["additionalProperties"] is False
    assert set(search_schema["required"]) == {"query", "jurisdiction"}
    assert search_schema["properties"]["max_results"]["maximum"] == 10


def test_search_tool_is_not_allow_listed_without_a_provider() -> None:
    with pytest.raises(ToolValidationError) as error:
        build_tool_registry().validate_call(
            tool_name="search_official_sources",
            arguments=search_arguments(),
        )

    assert error.value.code == "tool_not_allowed"


def test_only_https_results_from_configured_official_domains_are_returned() -> None:
    provider = StubOfficialSearchProvider(
        [
            OfficialSearchHit(
                title="Seva Sindhu",
                url="https://sevasindhu.karnataka.gov.in/service",
                authority="Government of Karnataka",
                snippet="Apply for the certificate online.",
            ),
            OfficialSearchHit(
                title="Official directory",
                url="https://district.nic.in/certificate",
                authority="District Administration",
                snippet="Certificate office details.",
            ),
            OfficialSearchHit(
                title="Impersonating site",
                url="https://karnataka.gov.in.example.com/fake",
                authority="Unknown",
                snippet="This must be filtered.",
            ),
            OfficialSearchHit(
                title="Insecure official URL",
                url="http://service.gov.in/insecure",
                authority="Unknown",
                snippet="HTTP must be filtered.",
            ),
        ]
    )
    registry = build_tool_registry(official_search_provider=provider)
    executor = ToolExecutor(registry)
    proposal = executor.prepare(
        actor_id="session-1",
        tool_name="search_official_sources",
        arguments=search_arguments(),
    )

    receipt = asyncio.run(
        executor.execute(actor_id="session-1", proposal_id=proposal.proposal_id)
    )

    assert receipt.result.ok is True
    assert receipt.result.summary == "Found 2 verified official result(s)."
    urls = [result["url"] for result in receipt.result.data["results"]]
    assert urls == [
        "https://sevasindhu.karnataka.gov.in/service",
        "https://district.nic.in/certificate",
    ]
    assert provider.calls == [
        {
            "query": "income certificate application",
            "jurisdiction": "Karnataka, India",
            "language": "English",
            "limit": 2,
        }
    ]


def test_no_verified_results_returns_a_safe_empty_result() -> None:
    provider = StubOfficialSearchProvider(
        [
            OfficialSearchHit(
                title="Unofficial blog",
                url="https://example.com/advice",
                authority="Blog",
                snippet="Unofficial advice.",
            )
        ]
    )
    tool = OfficialSearchTool(provider)

    result = asyncio.run(
        tool.execute(tool.argument_model.model_validate(search_arguments()))
    )

    assert result.ok is False
    assert result.data == {"results": []}


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "x", "jurisdiction": "India"},
        {"query": "certificate", "jurisdiction": "x"},
        {"query": "certificate", "jurisdiction": "India", "max_results": 11},
        {"query": "certificate", "jurisdiction": "India", "extra": True},
    ],
)
def test_search_arguments_are_validated_before_provider_call(
    arguments: dict[str, Any],
) -> None:
    provider = StubOfficialSearchProvider([])
    registry = build_tool_registry(official_search_provider=provider)

    with pytest.raises(ToolValidationError) as error:
        registry.validate_call(
            tool_name="search_official_sources",
            arguments=arguments,
        )

    assert error.value.code == "invalid_tool_arguments"
    assert provider.calls == []
