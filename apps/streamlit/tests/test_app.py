from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).parents[1] / "app.py"


def load_app() -> AppTest:
    return AppTest.from_file(str(APP_FILE), default_timeout=10).run()


def test_shell_renders_required_controls() -> None:
    app = load_app()

    assert not app.exception
    assert app.title[0].value == "Understand your document"
    assert [selectbox.label for selectbox in app.selectbox] == [
        "Language",
        "Explanation style",
    ]
    assert len(app.file_uploader) == 1
    assert app.file_uploader[0].label == "Upload an image or PDF"
    assert len(app.chat_input) == 1
    assert app.button[0].label == "Reset session"
    assert app.session_state["active_document"] is None
    assert app.session_state["extracted_facts"] == []


def test_chat_message_uses_selected_preferences() -> None:
    app = load_app()
    app.selectbox[0].select("Bengali")
    app.selectbox[1].select("Step by step")
    app.run()
    app.chat_input[0].set_value("What should I do next?").run()

    messages = app.session_state["messages"]
    assert messages[-2] == {"role": "user", "content": "What should I do next?"}
    assert "step by step" in messages[-1]["content"]
    assert "Bengali" in messages[-1]["content"]


def test_pdf_upload_shows_ready_state() -> None:
    app = load_app()
    app.file_uploader[0].upload(
        "notice.pdf",
        b"%PDF-1.4\n% minimal test document",
        "application/pdf",
    ).run()

    assert not app.exception
    assert app.success[0].value == "Ready: notice.pdf"
    assert any("PDF selected" in caption.value for caption in app.caption)
    assert app.session_state["active_document"]["filename"] == "notice.pdf"
    assert app.session_state["active_document"]["content_type"] == "application/pdf"


def test_extracted_facts_render_from_session_memory() -> None:
    app = load_app()
    app.file_uploader[0].upload(
        "bill.pdf",
        b"%PDF-1.4\n% bill",
        "application/pdf",
    ).run()
    app.session_state["extracted_facts"] = [
        {
            "label": "Due amount",
            "value": "₹1,250",
            "page": 1,
            "confidence": 0.96,
            "evidence": "Amount payable: ₹1,250",
        }
    ]
    app.run()

    assert any("Due amount" in markdown.value for markdown in app.markdown)
    assert any("96% confidence" in caption.value for caption in app.caption)


def test_reset_button_clears_document_facts_and_chat() -> None:
    app = load_app()
    app.file_uploader[0].upload(
        "bill.pdf",
        b"%PDF-1.4\n% bill",
        "application/pdf",
    ).run()
    app.session_state["extracted_facts"] = [
        {"label": "Due date", "value": "10 July"}
    ]
    app.chat_input[0].set_value("What is due?").run()

    app.button[0].click().run()

    assert app.session_state["active_document"] is None
    assert app.session_state["extracted_facts"] == []
    assert len(app.session_state["messages"]) == 1
    assert app.file_uploader[0].value is None
