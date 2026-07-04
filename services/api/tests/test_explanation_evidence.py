from app.explanation.evidence import (
    MAX_PAGE_QUOTE_CHARACTERS,
    build_source_evidence,
)
from app.schemas.explanation import EvidenceKind
from app.schemas.ocr import (
    OcrAmount,
    OcrDate,
    OcrDocumentResult,
    OcrEvidence,
    OcrPageResult,
    OcrSource,
)


def ocr_result() -> OcrDocumentResult:
    page = OcrPageResult(
        page=1,
        source=OcrSource.IMAGE,
        text="Bill date: 5 July 2026\nTotal due: ₹1,250",
        evidence=[
            OcrEvidence(
                page=1,
                label="Account status",
                value="Payment due",
                evidence_text="Total due: ₹1,250",
                confidence=0.92,
                grounded=True,
            ),
            OcrEvidence(
                page=1,
                label="Unsupported claim",
                value="Service will end tomorrow",
                evidence_text="not present",
                confidence=0.25,
                grounded=False,
            ),
        ],
        dates=[
            OcrDate(
                page=1,
                value="5 July 2026",
                normalized_value="2026-07-05",
                evidence_text="Bill date: 5 July 2026",
                confidence=0.9,
                grounded=True,
            )
        ],
        amounts=[
            OcrAmount(
                page=1,
                value="₹1,250",
                normalized_value="1250",
                currency="INR",
                evidence_text="Total due: ₹1,250",
                confidence=0.94,
                grounded=True,
            )
        ],
        confidence=0.9,
    )
    return OcrDocumentResult(
        page_count=1,
        text="--- Page 1 ---\n" + page.text,
        pages=[page],
        evidence=page.evidence,
        dates=page.dates,
        amounts=page.amounts,
        confidence=0.9,
    )


def test_catalog_has_stable_page_and_extracted_evidence_ids() -> None:
    catalog = build_source_evidence(ocr_result())

    assert [item.evidence_id for item in catalog] == ["P1", "E1", "E2", "E3"]
    assert [item.kind for item in catalog] == [
        EvidenceKind.PAGE_TEXT,
        EvidenceKind.FACT,
        EvidenceKind.DATE,
        EvidenceKind.AMOUNT,
    ]
    assert all(item.page == 1 for item in catalog)
    assert catalog[2].normalized_value == "2026-07-05"
    assert catalog[3].normalized_value == "1250"
    assert catalog[3].currency == "INR"


def test_ungrounded_ocr_items_are_not_citable() -> None:
    catalog = build_source_evidence(ocr_result())

    assert all(item.label != "Unsupported claim" for item in catalog)
    assert all("not present" not in item.quote for item in catalog)


def test_duplicate_extracted_items_are_collapsed() -> None:
    ocr = ocr_result()
    ocr.evidence.append(ocr.evidence[0].model_copy())

    catalog = build_source_evidence(ocr)

    fact_entries = [item for item in catalog if item.kind is EvidenceKind.FACT]
    assert len(fact_entries) == 1


def test_page_quotes_are_cleaned_and_bounded() -> None:
    ocr = ocr_result()
    ocr.pages[0].text = "  first\n\nline  " + ("x" * 2_000)

    catalog = build_source_evidence(ocr)

    assert catalog[0].quote.startswith("first line")
    assert len(catalog[0].quote) == MAX_PAGE_QUOTE_CHARACTERS
