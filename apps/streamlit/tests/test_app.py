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
