from pathlib import Path

import streamlit as st

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
        "Upload an image or PDF",
        type=("png", "jpg", "jpeg", "pdf"),
        key=upload_widget_key(st.session_state),
        help="Supported formats: PNG, JPG, JPEG, and PDF. Maximum size: 20 MB.",
    )

    if uploaded_file is None:
        if st.session_state.active_document is not None:
            clear_active_document(st.session_state)
        st.info("Choose a clear photo or PDF to begin.", icon="📄")
        return

    content = uploaded_file.getvalue()
    activate_document(
        st.session_state,
        filename=uploaded_file.name,
        content_type=uploaded_file.type,
        content=content,
    )

    st.success(f"Ready: {uploaded_file.name}", icon="✅")
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        st.image(uploaded_file, caption="Document preview", width="stretch")
    else:
        st.caption(f"PDF selected · {format_file_size(len(content))}")


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
            "- Prescriptions and acknowledgements"
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
