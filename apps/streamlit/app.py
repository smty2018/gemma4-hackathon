from pathlib import Path

import streamlit as st

from ui_config import LANGUAGES, style_instruction, style_labels


WELCOME_MESSAGE = (
    "Upload an image or PDF, then ask what it means. "
    "Document understanding will be connected in the next feature step."
)


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE},
        ]


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
        st.caption("Files are held only for the current session in this UI shell.")
    return language, explanation_style


def render_upload_area() -> None:
    st.subheader("1. Add a document")
    uploaded_file = st.file_uploader(
        "Upload an image or PDF",
        type=("png", "jpg", "jpeg", "pdf"),
        key="document_upload",
        help="Supported formats: PNG, JPG, JPEG, and PDF. Maximum size: 20 MB.",
    )

    if uploaded_file is None:
        st.info("Choose a clear photo or PDF to begin.", icon="📄")
        return

    st.success(f"Ready: {uploaded_file.name}", icon="✅")
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        st.image(uploaded_file, caption="Document preview", width="stretch")
    else:
        size_kb = len(uploaded_file.getvalue()) / 1024
        st.caption(f"PDF selected · {size_kb:.1f} KB")


def render_chat(language: str, explanation_style: str) -> None:
    st.subheader("2. Ask about the document")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask what this document means…", key="chat_prompt")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    response = (
        f"Your question is queued for a {explanation_style.lower()} response "
        f"in {language}. Gemma integration is the next implementation step."
    )
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Document Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_session()
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
    render_chat(language, explanation_style)


if __name__ == "__main__":
    main()
