import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import requests
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
    voice_widget_key,
)
from ui_config import (
    LANGUAGES,
    STYLE_KEYS,
    language_name,
    style_instruction,
    style_label,
    text,
)


TEXT_DISPLAY_CHARACTERS = 10_000
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SPEECH_LANGUAGE_CODES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",
}


def hide_streamlit_chrome() -> None:
    st.markdown(
        """
        <style>
        html body [data-testid="stToolbar"],
        html body [data-testid="stToolbarActions"],
        html body [data-testid="stAppDeployButton"],
        html body [data-testid="stDecoration"],
        html body [data-testid="stStatusWidget"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def localize_file_uploader(language: str) -> None:
    button_label = text(language, "upload_button").replace('"', "&quot;")
    st.markdown(
        f"""
        <style>
        html body [data-testid="stFileUploaderDropzoneInstructions"],
        html body [data-testid="stFileUploaderDropzone"] small {{
            display: none !important;
        }}
        html body [data-testid="stFileUploaderDropzone"] button {{
            font-size: 0 !important;
        }}
        html body [data-testid="stFileUploaderDropzone"] button::after {{
            content: "{button_label}";
            font-size: 0.875rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_ingestion(
    filename: str, content_type: str | None, content: bytes
) -> IngestionResult:
    return ingest_upload(
        filename=filename,
        content_type=content_type,
        content=content,
    )


@st.cache_data(show_spinner=False, ttl=3_600, max_entries=16)
def cached_speech(text_value: str, language: str) -> bytes:
    payload = json.dumps(
        {
            "text": text_value,
            "target_language_code": SPEECH_LANGUAGE_CODES[language],
        }
    ).encode("utf-8")
    request = Request(
        f"{API_BASE_URL}/api/v1/tts/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            audio = response.read()
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError("Speech service unavailable") from error
    if not audio:
        raise RuntimeError("Speech service returned no audio")
    return audio


@st.cache_data(show_spinner=False, ttl=3_600, max_entries=32)
def cached_transcription(
    content: bytes,
    content_type: str,
    language: str,
) -> dict[str, object]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/stt/transcribe",
            files={"file": ("voice-question.wav", content, content_type)},
            data={"language_code": SPEECH_LANGUAGE_CODES[language]},
            timeout=90,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError("Speech transcription unavailable") from error

    transcript = str(result.get("transcript", "")).strip()
    if not transcript:
        raise RuntimeError("Speech service returned no transcript")
    return {
        "transcript": transcript,
        "language_code": result.get("language_code"),
        "language_probability": result.get("language_probability"),
    }


def render_language_gate() -> None:
    st.title("Choose your language · अपनी भाषा चुनें · আপনার ভাষা নির্বাচন করুন")
    st.write("Select a language · भाषा चुनें · ভাষা নির্বাচন করুন")
    language = st.selectbox(
        "Language · भाषा · ভাষা",
        LANGUAGES,
        format_func=language_name,
        key="language_choice",
    )
    if st.button(
        "Continue · आगे बढ़ें · এগিয়ে যান",
        type="primary",
        use_container_width=True,
        key="confirm_language",
    ):
        st.session_state.ui_language = language
        st.session_state.language_confirmed = True
        reset_memory(
            st.session_state,
            welcome_message=text(language, "welcome"),
        )
        st.rerun()


def render_sidebar(language: str) -> str:
    with st.sidebar:
        st.header(text(language, "response_settings"))
        st.caption(
            f"{text(language, 'selected_language')}: {language_name(language)}"
        )
        if st.button(
            text(language, "change_language"),
            key="change_language",
            use_container_width=True,
        ):
            reset_memory(
                st.session_state,
                welcome_message=text(language, "welcome"),
            )
            st.session_state.language_confirmed = False
            st.session_state.ui_language = None
            st.rerun()

        explanation_style = st.selectbox(
            text(language, "explanation_style"),
            STYLE_KEYS,
            format_func=lambda key: style_label(key, language),
            key=f"explanation_style_{language}",
            help=text(language, "explanation_style_help"),
        )
        st.caption(style_instruction(explanation_style, language))
        st.divider()

        st.subheader(text(language, "current_session"))
        active_document = st.session_state.active_document
        if active_document:
            st.markdown(f"**{active_document['filename']}**")
            st.caption(
                f"{active_document['content_type']} · "
                f"{format_file_size(active_document['size_bytes'])}"
            )
            st.caption(
                text(
                    language,
                    "session_counts",
                    facts=len(st.session_state.extracted_facts),
                    messages=len(st.session_state.messages),
                )
            )
        else:
            st.caption(text(language, "no_active_document"))

        if st.button(
            text(language, "reset_session"),
            key="reset_session",
            use_container_width=True,
            help=text(language, "reset_help"),
        ):
            reset_memory(
                st.session_state,
                welcome_message=text(language, "welcome"),
            )
            st.rerun()

        st.divider()
        st.caption(text(language, "session_privacy"))
    return explanation_style


def render_upload_area(language: str) -> None:
    st.subheader(text(language, "add_document"))
    localize_file_uploader(language)
    uploaded_file = st.file_uploader(
        text(language, "upload_label"),
        type=("png", "jpg", "jpeg", "pdf", "wav", "mp3", "m4a"),
        key=upload_widget_key(st.session_state),
        help=text(language, "upload_help"),
    )
    st.caption(text(language, "upload_limits"))

    if uploaded_file is None:
        if st.session_state.active_document is not None:
            clear_active_document(
                st.session_state,
                welcome_message=text(language, "welcome"),
            )
        st.info(text(language, "choose_file"), icon="📄")
        return

    content = uploaded_file.getvalue()
    try:
        with st.spinner(text(language, "validating")):
            result = cached_ingestion(
                uploaded_file.name,
                uploaded_file.type,
                content,
            )
    except (UploadValidationError, PreviewGenerationError):
        clear_active_document(
            st.session_state,
            welcome_message=text(language, "welcome"),
        )
        st.error(text(language, "upload_error"), icon="⚠️")
        return

    activate_document(
        st.session_state,
        filename=result.descriptor.filename,
        content_type=result.descriptor.content_type,
        content=content,
        welcome_message=text(language, "welcome"),
        active_document_message=text(
            language,
            "document_active",
            filename=result.descriptor.filename,
        ),
    )

    st.success(
        text(language, "ready", filename=result.descriptor.filename),
        icon="✅",
    )
    st.caption(
        f"{text(language, f'kind_{result.descriptor.kind.value}')} · "
        f"{format_file_size(result.descriptor.size_bytes)}"
    )
    render_file_preview(result, language)


def render_file_preview(result: IngestionResult, language: str) -> None:
    if result.image_preview:
        preview = result.image_preview
        st.image(
            preview.content,
            caption=text(
                language,
                "image_preview",
                source_width=preview.source_width,
                source_height=preview.source_height,
                width=preview.width,
                height=preview.height,
            ),
            width="stretch",
        )
        return

    if result.pdf_preview:
        preview = result.pdf_preview
        shown_pages = len(preview.pages)
        suffix = text(language, "additional_pages") if preview.truncated else ""
        st.caption(
            text(
                language,
                "pdf_preview",
                pages=preview.page_count,
                shown=shown_pages,
                suffix=suffix,
            )
        )
        columns = st.columns(2)
        for index, page in enumerate(preview.pages):
            with columns[index % len(columns)]:
                st.image(
                    page.content,
                    caption=text(language, "page", number=page.page_number),
                    width="stretch",
                )
        render_pdf_content(result, language)
        return

    if result.audio_preview:
        preview = result.audio_preview
        st.audio(preview.content, format=preview.content_type)
        details = [text(language, "seconds", value=preview.duration_seconds)]
        if preview.sample_rate:
            details.append(f"{preview.sample_rate:,} Hz")
        if preview.channels:
            channel_key = "channel" if preview.channels == 1 else "channels"
            details.append(f"{preview.channels} {text(language, channel_key)}")
        st.caption(" · ".join(details))


def render_pdf_content(result: IngestionResult, language: str) -> None:
    content = result.pdf_content
    if content is None:
        return

    text_page_key = "page_single" if content.text_page_count == 1 else "page_plural"
    scanned_page_count = len(content.scanned_page_numbers)
    scanned_page_key = "page_single" if scanned_page_count == 1 else "page_plural"
    st.markdown(f"#### {text(language, 'pdf_processing')}")
    st.caption(
        text(
            language,
            "pdf_counts",
            text_count=content.text_page_count,
            text_label=text(language, text_page_key),
            scan_count=scanned_page_count,
            scan_label=text(language, scanned_page_key),
        )
    )

    if content.embedded_text:
        displayed_text = content.embedded_text[:TEXT_DISPLAY_CHARACTERS]
        display_truncated = len(content.embedded_text) > len(displayed_text)
        st.text_area(
            text(language, "embedded_text"),
            value=displayed_text,
            height=220,
            disabled=True,
        )
        if content.text_truncated or display_truncated:
            st.caption(text(language, "text_truncated"))
    else:
        st.info(text(language, "no_embedded_text"))

    if not content.scanned_page_numbers:
        st.caption(text(language, "no_scanned_pages"))
        return

    st.markdown(f"#### {text(language, 'scanned_ready')}")
    st.caption(
        text(
            language,
            "detected_pages",
            pages=", ".join(str(page) for page in content.scanned_page_numbers),
        )
    )
    columns = st.columns(2)
    for index, page in enumerate(content.scanned_pages):
        with columns[index % len(columns)]:
            st.image(
                page.content,
                caption=text(language, "scanned_page", number=page.page_number),
                width="stretch",
            )
    if content.scanned_previews_truncated:
        st.caption(text(language, "scanned_truncated"))


def render_extracted_facts(language: str) -> None:
    st.subheader(text(language, "session_memory"))
    active_document = st.session_state.active_document
    facts = st.session_state.extracted_facts

    if active_document is None:
        st.caption(text(language, "active_none"))
        st.info(text(language, "upload_for_memory"))
        return

    document_column, fact_column, message_column = st.columns(3)
    document_column.metric(
        text(language, "active_document"), active_document["filename"]
    )
    fact_column.metric(text(language, "extracted_facts"), len(facts))
    message_column.metric(text(language, "messages"), len(st.session_state.messages))

    st.markdown(f"#### {text(language, 'extracted_facts')}")
    if not facts:
        st.info(text(language, "no_facts"))
        return

    for fact in facts:
        st.markdown(f"**{fact['label']}:** {fact['value']}")
        details: list[str] = []
        if fact.get("page") is not None:
            details.append(text(language, "fact_page", number=fact["page"]))
        if fact.get("confidence") is not None:
            details.append(text(language, "confidence", value=fact["confidence"]))
        if fact.get("evidence"):
            details.append(text(language, "evidence", value=fact["evidence"]))
        if details:
            st.caption(" · ".join(details))


def render_chat(language: str, explanation_style: str) -> None:
    st.subheader(text(language, "ask_document"))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(text(language, "chat_placeholder"), key="chat_prompt")
    if not prompt:
        return

    queue_question(language, explanation_style, prompt)
    st.rerun()


def queue_question(language: str, explanation_style: str, prompt: str) -> None:
    add_message(st.session_state, "user", prompt)
    add_message(
        st.session_state,
        "assistant",
        text(
            language,
            "queued_response",
            style=style_label(explanation_style, language),
        ),
    )


def latest_assistant_reply() -> str | None:
    for message in reversed(st.session_state.messages[1:]):
        if message["role"] == "assistant" and message["content"].strip():
            return message["content"]
    return None


def render_voice_ui(language: str, explanation_style: str) -> None:
    st.subheader(text(language, "voice_title"))
    st.caption(text(language, "voice_intro"))

    input_column, output_column = st.columns(2, gap="large")
    with input_column:
        recording = st.audio_input(
            text(language, "record_label"),
            sample_rate=16_000,
            key=voice_widget_key(st.session_state),
            help=text(language, "record_help"),
        )
        if recording is not None:
            content = recording.getvalue()
            try:
                result = cached_ingestion(
                    "voice-question.wav",
                    recording.type or "audio/wav",
                    content,
                )
            except (UploadValidationError, PreviewGenerationError):
                st.session_state.voice_question = None
                st.error(text(language, "voice_error"), icon="⚠️")
            else:
                preview = result.audio_preview
                st.session_state.voice_question = {
                    "content": content,
                    "content_type": recording.type or "audio/wav",
                    "duration_seconds": (
                        preview.duration_seconds if preview is not None else None
                    ),
                }
                st.success(text(language, "recording_ready"), icon="🎙️")
                try:
                    with st.spinner(text(language, "transcribing")):
                        transcription = cached_transcription(
                            content,
                            recording.type or "audio/wav",
                            language,
                        )
                except RuntimeError:
                    st.error(text(language, "stt_error"), icon="📝")
                else:
                    transcript = st.text_area(
                        text(language, "transcript_label"),
                        value=str(transcription["transcript"]),
                        height=100,
                        key=f"voice_transcript_{result.descriptor.content_hash[:12]}",
                    )
                    detected_language = transcription.get("language_code")
                    if detected_language:
                        st.caption(
                            text(
                                language,
                                "detected_language",
                                language_code=detected_language,
                            )
                        )
                    st.session_state.voice_question.update(transcription)
                    if st.button(
                        text(language, "use_voice_question"),
                        key="use_voice_question",
                        type="primary",
                        use_container_width=True,
                    ):
                        if transcript.strip():
                            queue_question(language, explanation_style, transcript)
                            st.rerun()
                        else:
                            st.error(text(language, "transcript_empty"))

    with output_column:
        st.markdown(f"#### {text(language, 'voice_reply_title')}")
        st.caption(text(language, "voice_reply_help"))
        reply = latest_assistant_reply()
        play_reply = st.button(
            text(language, "hear_latest"),
            key="hear_latest_reply",
            disabled=reply is None,
            use_container_width=True,
        )
        if reply is None:
            st.info(text(language, "no_reply"))
        elif play_reply:
            try:
                with st.spinner(text(language, "speaking")):
                    speech = cached_speech(reply, language)
            except RuntimeError:
                st.error(text(language, "speech_error"), icon="🔊")
            else:
                st.audio(speech, format="audio/mpeg", autoplay=True)


def main() -> None:
    selected_language = st.session_state.get("ui_language")
    page_title = (
        text(selected_language, "page_title")
        if selected_language in LANGUAGES
        else "Choose language"
    )
    st.set_page_config(
        page_title=page_title,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    hide_streamlit_chrome()
    st.session_state.setdefault("language_confirmed", False)
    st.session_state.setdefault("ui_language", None)

    if not st.session_state.language_confirmed or selected_language not in LANGUAGES:
        render_language_gate()
        return

    language = selected_language
    initialize_memory(
        st.session_state,
        welcome_message=text(language, "welcome"),
    )
    explanation_style = render_sidebar(language)

    st.title(text(language, "title"))
    st.write(text(language, "intro"))

    upload_column, guidance_column = st.columns((1.5, 1), gap="large")
    with upload_column:
        render_upload_area(language)
    with guidance_column:
        st.subheader(text(language, "what_add"))
        st.markdown(text(language, "guidance"))
        st.warning(text(language, "privacy_warning"), icon="🔒")

    st.divider()
    render_extracted_facts(language)
    st.divider()
    render_voice_ui(language, explanation_style)
    st.divider()
    render_chat(language, explanation_style)


if __name__ == "__main__":
    main()
