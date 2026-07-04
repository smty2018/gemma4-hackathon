import pytest

from session_memory import (
    WELCOME_MESSAGE,
    activate_document,
    add_message,
    clear_active_document,
    format_file_size,
    initialize_memory,
    replace_extracted_facts,
    reset_memory,
    upload_widget_key,
)


def fresh_state() -> dict:
    state: dict = {}
    initialize_memory(state)
    return state


def test_initialize_memory_sets_all_document_scoped_defaults() -> None:
    state = fresh_state()

    assert state["messages"] == [{"role": "assistant", "content": WELCOME_MESSAGE}]
    assert state["active_document"] is None
    assert state["extracted_facts"] == []
    assert upload_widget_key(state) == "document_upload_0"


def test_activate_document_stores_safe_metadata_and_resets_context() -> None:
    state = fresh_state()
    add_message(state, "user", "Old question")
    state["extracted_facts"] = [{"label": "Old", "value": "Fact"}]

    changed = activate_document(
        state,
        filename="../notice.pdf",
        content_type="application/pdf",
        content=b"document bytes",
    )

    assert changed is True
    assert state["active_document"]["filename"] == "notice.pdf"
    assert state["active_document"]["extension"] == ".pdf"
    assert state["active_document"]["size_bytes"] == 14
    assert len(state["active_document"]["document_id"]) == 16
    assert state["extracted_facts"] == []
    assert "Old question" not in [message["content"] for message in state["messages"]]


def test_reselecting_same_document_preserves_memory() -> None:
    state = fresh_state()
    arguments = {
        "filename": "bill.pdf",
        "content_type": "application/pdf",
        "content": b"same document",
    }
    activate_document(state, **arguments)
    state["extracted_facts"] = [{"label": "Due date", "value": "10 July"}]
    add_message(state, "user", "Keep this question")

    changed = activate_document(state, **arguments)

    assert changed is False
    assert state["extracted_facts"][0]["label"] == "Due date"
    assert state["messages"][-1]["content"] == "Keep this question"


def test_extracted_facts_require_an_active_document() -> None:
    state = fresh_state()

    with pytest.raises(ValueError, match="active document"):
        replace_extracted_facts(state, [{"label": "Amount", "value": "100"}])


def test_clear_active_document_removes_document_scoped_memory() -> None:
    state = fresh_state()
    activate_document(
        state,
        filename="bill.png",
        content_type="image/png",
        content=b"image",
    )
    replace_extracted_facts(state, [{"label": "Amount", "value": "100"}])

    clear_active_document(state)

    assert state["active_document"] is None
    assert state["extracted_facts"] == []
    assert state["messages"] == [{"role": "assistant", "content": WELCOME_MESSAGE}]


def test_reset_memory_clears_state_and_rotates_upload_widget() -> None:
    state = fresh_state()
    activate_document(
        state,
        filename="bill.pdf",
        content_type="application/pdf",
        content=b"document",
    )

    reset_memory(state)

    assert state["active_document"] is None
    assert state["extracted_facts"] == []
    assert state["messages"] == [{"role": "assistant", "content": WELCOME_MESSAGE}]
    assert upload_widget_key(state) == "document_upload_1"


@pytest.mark.parametrize(
    ("size_bytes", "formatted"),
    [(512, "512 B"), (2048, "2.0 KB"), (2 * 1024 * 1024, "2.0 MB")],
)
def test_format_file_size(size_bytes: int, formatted: str) -> None:
    assert format_file_size(size_bytes) == formatted
