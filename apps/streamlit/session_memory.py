from hashlib import sha256
from pathlib import Path
from typing import Any, MutableMapping, TypedDict


WELCOME_MESSAGE = (
    "Upload an image, PDF, or audio recording, then ask what it means. "
    "Document understanding will be connected in the next feature step."
)


class ChatMessage(TypedDict):
    role: str
    content: str


class ActiveDocument(TypedDict):
    document_id: str
    filename: str
    content_type: str
    extension: str
    size_bytes: int


class ExtractedFact(TypedDict, total=False):
    label: str
    value: str
    evidence: str
    page: int
    confidence: float


SessionState = MutableMapping[str, Any]


def initial_messages() -> list[ChatMessage]:
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def initialize_memory(state: SessionState) -> None:
    state.setdefault("messages", initial_messages())
    state.setdefault("active_document", None)
    state.setdefault("extracted_facts", [])
    state.setdefault("tool_results", [])
    state.setdefault("upload_revision", 0)


def upload_widget_key(state: SessionState) -> str:
    return f"document_upload_{state['upload_revision']}"


def add_message(state: SessionState, role: str, content: str) -> ChatMessage:
    if role not in {"assistant", "user"}:
        raise ValueError(f"Unsupported chat role: {role}")
    if not content.strip():
        raise ValueError("Chat messages cannot be empty")

    message: ChatMessage = {"role": role, "content": content.strip()}
    state["messages"].append(message)
    return message


def build_document_record(
    *, filename: str, content_type: str, content: bytes
) -> ActiveDocument:
    safe_filename = Path(filename).name or "document"
    return {
        "document_id": sha256(content).hexdigest()[:16],
        "filename": safe_filename,
        "content_type": content_type or "application/octet-stream",
        "extension": Path(safe_filename).suffix.lower(),
        "size_bytes": len(content),
    }


def activate_document(
    state: SessionState,
    *,
    filename: str,
    content_type: str,
    content: bytes,
) -> bool:
    """Set the active document and return whether the document changed."""

    document = build_document_record(
        filename=filename,
        content_type=content_type,
        content=content,
    )
    current = state.get("active_document")
    if current and current["document_id"] == document["document_id"]:
        return False

    state["active_document"] = document
    state["extracted_facts"] = []
    state["tool_results"] = []
    state["messages"] = initial_messages()
    add_message(
        state,
        "assistant",
        f"{document['filename']} is now the active document. What would you like to know?",
    )
    return True


def clear_active_document(state: SessionState) -> None:
    state["active_document"] = None
    state["extracted_facts"] = []
    state["tool_results"] = []
    state["messages"] = initial_messages()


def replace_extracted_facts(
    state: SessionState, facts: list[ExtractedFact]
) -> None:
    if state.get("active_document") is None and facts:
        raise ValueError("Cannot store extracted facts without an active document")
    state["extracted_facts"] = [dict(fact) for fact in facts]


def reset_memory(state: SessionState) -> None:
    state["messages"] = initial_messages()
    state["active_document"] = None
    state["extracted_facts"] = []
    state["tool_results"] = []
    state["upload_revision"] = int(state.get("upload_revision", 0)) + 1


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"
