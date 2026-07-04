import streamlit as st

from ingestion import (
    IngestionResult,
    PreviewGenerationError,
    UploadValidationError,
    ingest_upload,
)
from session_memory import (
    activate_document,
    add_message,
    clear_active_document,
    format_file_size,
    initialize_memory,
    reset_memory,
    upload_widget_key,
)
from ui_config import LANGUAGES, style_instruction, style_labels


TEXT_DISPLAY_CHARACTERS = 10_000


@st.cache_data(show_spinner=False, max_entries=8)
def cached_ingestion(
    filename: str, content_type: str | None, content: bytes
) -> IngestionResult:
    return ingest_upload(
        filename=filename,
        content_type=content_type,
        content=content,
    )


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.header("Response settings")
        language = st.selectbox(
            "Language",
            LANGUAGES,
            key="language",
            help="The assistant will respond in this language.",
        )
        explanation_style = st.selectbox(
            "Explanation style",
            style_labels(),
            key="explanation_style",
            help="Choose how detailed and audience-aware the explanation should be.",
        )
        st.caption(style_instruction(explanation_style))
        st.divider()

        st.subheader("Current session")
        active_document = st.session_state.active_document
        if active_document:
            st.markdown(f"**{active_document['filename']}**")
            st.caption(
                f"{active_document['content_type']} · "
                f"{format_file_size(active_document['size_bytes'])}"
            )
            st.caption(
                f"{len(st.session_state.extracted_facts)} extracted facts · "
                f"{len(st.session_state.messages)} messages"
            )
        else:
            st.caption("No active document")

        if st.button(
            "Reset session",
            key="reset_session",
            use_container_width=True,
            help="Clear the active document, extracted facts, and conversation.",
        ):
            reset_memory(st.session_state)
            st.rerun()

        st.divider()
        st.caption("Files are held only for the current session in this UI shell.")
    return language, explanation_style


def render_upload_area() -> None:
    st.subheader("1. Add a document")
    uploaded_file = st.file_uploader(
        "Upload an image, PDF, or audio recording",
        type=("png", "jpg", "jpeg", "pdf", "wav", "mp3", "m4a"),
        key=upload_widget_key(st.session_state),
        help=(
            "PNG/JPEG up to 10 MB, PDF up to 20 MB, "
            "or WAV/MP3/M4A up to 15 MB and 30 seconds."
        ),
    )

    if uploaded_file is None:
        if st.session_state.active_document is not None:
            clear_active_document(st.session_state)
        st.info("Choose a clear image, PDF, or short audio recording to begin.", icon="📄")
        return

    content = uploaded_file.getvalue()
    try:
        with st.spinner("Validating file and preparing preview…"):
            result = cached_ingestion(
                uploaded_file.name,
                uploaded_file.type,
                content,
            )
    except (UploadValidationError, PreviewGenerationError) as error:
        clear_active_document(st.session_state)
        st.error(str(error), icon="⚠️")
        return

    activate_document(
        st.session_state,
        filename=result.descriptor.filename,
        content_type=result.descriptor.content_type,
        content=content,
    )

    st.success(f"Ready: {result.descriptor.filename}", icon="✅")
    st.caption(
        f"{result.descriptor.kind.value.title()} · "
        f"{format_file_size(result.descriptor.size_bytes)}"
    )
    render_file_preview(result)


def render_file_preview(result: IngestionResult) -> None:
    if result.image_preview:
        preview = result.image_preview
        st.image(
            preview.content,
            caption=(
                f"Image preview · {preview.source_width}×{preview.source_height} source · "
                f"{preview.width}×{preview.height} preview"
            ),
            width="stretch",
        )
        return

    if result.pdf_preview:
        preview = result.pdf_preview
        shown_pages = len(preview.pages)
        suffix = " · additional pages not shown" if preview.truncated else ""
        st.caption(f"{preview.page_count} pages · previewing {shown_pages}{suffix}")
        columns = st.columns(2)
        for index, page in enumerate(preview.pages):
            with columns[index % len(columns)]:
                st.image(
                    page.content,
                    caption=f"Page {page.page_number}",
                    width="stretch",
                )
        render_pdf_content(result)
        return

    if result.audio_preview:
        preview = result.audio_preview
        st.audio(preview.content, format=preview.content_type)
        details = [f"{preview.duration_seconds:.1f} seconds"]
        if preview.sample_rate:
            details.append(f"{preview.sample_rate:,} Hz")
        if preview.channels:
            label = "channel" if preview.channels == 1 else "channels"
            details.append(f"{preview.channels} {label}")
        st.caption(" · ".join(details))


def render_pdf_content(result: IngestionResult) -> None:
    content = result.pdf_content
    if content is None:
        return

    text_page_label = "page" if content.text_page_count == 1 else "pages"
    scanned_page_count = len(content.scanned_page_numbers)
    scanned_page_label = "page" if scanned_page_count == 1 else "pages"
    st.markdown("#### PDF processing")
    st.caption(
        f"{content.text_page_count} text {text_page_label} · "
        f"{scanned_page_count} scanned {scanned_page_label}"
    )

    if content.embedded_text:
        displayed_text = content.embedded_text[:TEXT_DISPLAY_CHARACTERS]
        display_truncated = len(content.embedded_text) > len(displayed_text)
        st.text_area(
            "Extracted embedded text",
            value=displayed_text,
            height=220,
            disabled=True,
        )
        if content.text_truncated or display_truncated:
            st.caption("Extracted text is truncated to keep processing responsive.")
    else:
        st.info("No embedded text found. Scanned pages are ready for OCR.")

    if not content.scanned_page_numbers:
        st.caption("No scanned pages detected.")
        return

    st.markdown("#### Scanned pages prepared for OCR")
    st.caption(
        "Detected pages: "
        + ", ".join(str(page) for page in content.scanned_page_numbers)
    )
    columns = st.columns(2)
    for index, page in enumerate(content.scanned_pages):
        with columns[index % len(columns)]:
            st.image(
                page.content,
                caption=f"Scanned page {page.page_number}",
                width="stretch",
            )
    if content.scanned_previews_truncated:
        st.caption("Additional scanned pages were detected but are not rendered here.")


def render_extracted_facts() -> None:
    st.subheader("Session memory")
    active_document = st.session_state.active_document
    facts = st.session_state.extracted_facts

    if active_document is None:
        st.caption("Active document: none")
        st.info("Upload a document to create document-scoped memory.")
        return

    document_column, fact_column, message_column = st.columns(3)
    document_column.metric("Active document", active_document["filename"])
    fact_column.metric("Extracted facts", len(facts))
    message_column.metric("Messages", len(st.session_state.messages))

    st.markdown("#### Extracted facts")
    if not facts:
        st.info("No facts extracted yet. OCR and document analysis are the next feature step.")
        return

    for fact in facts:
        st.markdown(f"**{fact['label']}:** {fact['value']}")
        details: list[str] = []
        if fact.get("page") is not None:
            details.append(f"page {fact['page']}")
        if fact.get("confidence") is not None:
            details.append(f"{fact['confidence']:.0%} confidence")
        if fact.get("evidence"):
            details.append(f"evidence: {fact['evidence']}")
        if details:
            st.caption(" · ".join(details))


def render_chat(language: str, explanation_style: str) -> None:
    st.subheader("2. Ask about the document")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask what this document means…", key="chat_prompt")
    if not prompt:
        return

    add_message(st.session_state, "user", prompt)
    response = (
        f"Your question is queued for a {explanation_style.lower()} response "
        f"in {language}. Gemma integration is the next implementation step."
    )
    add_message(st.session_state, "assistant", response)
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Document Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_memory(st.session_state)
    language, explanation_style = render_sidebar()

    st.title("Understand your document")
    st.write(
        "Upload a document and ask questions in the language and explanation style "
        "that work best for you."
    )

    upload_column, guidance_column = st.columns((1.5, 1), gap="large")
    with upload_column:
        render_upload_area()
    with guidance_column:
        st.subheader("What you can add")
        st.markdown(
            "- Government notices\n"
            "- Electricity or property bills\n"
            "- Bank and insurance letters\n"
            "- Prescriptions and acknowledgements\n"
            "- Short voice questions"
        )
        st.warning(
            "Do not upload passwords, OTPs, or documents you do not have permission to use.",
            icon="🔒",
        )

    st.divider()
    render_extracted_facts()
    st.divider()
    render_chat(language, explanation_style)


if __name__ == "__main__":
    main()
