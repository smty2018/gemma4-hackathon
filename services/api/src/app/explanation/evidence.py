import re

from app.schemas.explanation import EvidenceKind, SourceEvidence
from app.schemas.ocr import OcrDocumentResult

MAX_PAGE_QUOTE_CHARACTERS = 1_000
MAX_SOURCE_EVIDENCE = 300


def build_source_evidence(ocr: OcrDocumentResult) -> list[SourceEvidence]:
    entries: list[SourceEvidence] = []

    for page in sorted(ocr.pages, key=lambda item: item.page):
        quote = _clean_quote(page.text)[:MAX_PAGE_QUOTE_CHARACTERS]
        if not quote:
            continue
        entries.append(
            SourceEvidence(
                evidence_id=f"P{page.page}",
                page=page.page,
                kind=EvidenceKind.PAGE_TEXT,
                label=f"Page {page.page} text",
                value=quote,
                quote=quote,
                confidence=page.confidence,
            )
        )

    seen: set[tuple[int, EvidenceKind, str, str]] = set()
    extracted_index = 1

    extracted_items = [
        *(
            (
                item.page,
                EvidenceKind.FACT,
                item.label,
                item.value,
                None,
                None,
                item.evidence_text,
                item.confidence,
                item.grounded,
            )
            for item in ocr.evidence
        ),
        *(
            (
                item.page,
                EvidenceKind.DATE,
                "Date",
                item.value,
                item.normalized_value,
                None,
                item.evidence_text,
                item.confidence,
                item.grounded,
            )
            for item in ocr.dates
        ),
        *(
            (
                item.page,
                EvidenceKind.AMOUNT,
                "Amount",
                item.value,
                item.normalized_value,
                item.currency,
                item.evidence_text,
                item.confidence,
                item.grounded,
            )
            for item in ocr.amounts
        ),
    ]

    for (
        page,
        kind,
        label,
        value,
        normalized_value,
        currency,
        evidence_text,
        confidence,
        grounded,
    ) in extracted_items:
        if not grounded or len(entries) >= MAX_SOURCE_EVIDENCE:
            continue
        quote = _clean_quote(evidence_text)
        deduplication_key = (page, kind, value.casefold(), quote.casefold())
        if deduplication_key in seen:
            continue
        seen.add(deduplication_key)

        entries.append(
            SourceEvidence(
                evidence_id=f"E{extracted_index}",
                page=page,
                kind=kind,
                label=label,
                value=value,
                normalized_value=normalized_value,
                currency=currency,
                quote=quote,
                confidence=confidence,
            )
        )
        extracted_index += 1

    return entries[:MAX_SOURCE_EVIDENCE]


def _clean_quote(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
