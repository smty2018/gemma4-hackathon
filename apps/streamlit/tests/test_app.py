from io import BytesIO
from pathlib import Path
import wave

import pymupdf
from PIL import Image
from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).parents[1] / "app.py"


def pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page(width=300, height=400).insert_text((30, 40), "Notice")
    content = document.tobytes()
    document.close()
    return content


def mixed_pdf_bytes() -> bytes:
    image_output = BytesIO()
    Image.new("RGB", (80, 60), color=(30, 90, 150)).save(
        image_output,
        format="PNG",
    )

    document = pymupdf.open()
    document.new_page(width=300, height=400).insert_text(
        (30, 40),
        "Embedded notice text",
    )
    scanned_page = document.new_page(width=300, height=400)
    scanned_page.insert_image(scanned_page.rect, stream=image_output.getvalue())
    content = document.tobytes()
    document.close()
    return content


def wav_bytes() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(8_000)
        recording.writeframes(b"\x00\x00" * 4_000)
    return output.getvalue()


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
    assert app.file_uploader[0].label == "Upload an image, PDF, or audio recording"
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
        pdf_bytes(),
        "application/pdf",
    ).run()

    assert not app.exception
    assert app.success[0].value == "Ready: notice.pdf"
    assert any("Pdf" in caption.value for caption in app.caption)
    assert app.session_state["active_document"]["filename"] == "notice.pdf"
    assert app.session_state["active_document"]["content_type"] == "application/pdf"
    assert any("1 pages" in caption.value for caption in app.caption)
    assert any("1 text page" in caption.value for caption in app.caption)
    assert app.text_area[0].label == "Extracted embedded text"
    assert "Notice" in app.text_area[0].value
    assert any("No scanned pages detected" in caption.value for caption in app.caption)


def test_mixed_pdf_marks_scanned_pages_for_ocr() -> None:
    app = load_app()
    app.file_uploader[0].upload(
        "mixed.pdf",
        mixed_pdf_bytes(),
        "application/pdf",
    ).run()

    assert not app.exception
    assert "Embedded notice text" in app.text_area[0].value
    assert any("1 scanned page" in caption.value for caption in app.caption)
    assert any("Detected pages: 2" in caption.value for caption in app.caption)
    assert any(
        "Scanned pages prepared for OCR" in markdown.value
        for markdown in app.markdown
    )


def test_extracted_facts_render_from_session_memory() -> None:
    app = load_app()
    app.file_uploader[0].upload(
        "bill.pdf",
        pdf_bytes(),
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
        pdf_bytes(),
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


def test_invalid_file_does_not_become_active_document() -> None:
    app = load_app()
    app.file_uploader[0].upload(
        "fake.pdf",
        b"not a pdf",
        "application/pdf",
    ).run()

    assert app.session_state["active_document"] is None
    assert any("do not match" in error.value for error in app.error)


def test_audio_upload_renders_playback_metadata() -> None:
    app = load_app()
    app.file_uploader[0].upload(
        "question.wav",
        wav_bytes(),
        "audio/wav",
    ).run()

    assert not app.exception
    assert app.session_state["active_document"]["content_type"] == "audio/wav"
    assert any("0.5 seconds" in caption.value for caption in app.caption)
