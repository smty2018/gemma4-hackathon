from dataclasses import dataclass


@dataclass(frozen=True)
class ExplanationStyle:
    label: str
    instruction: str


LANGUAGES: tuple[str, ...] = (
    "English",
    "Hindi",
    "Bengali",
    "Tamil",
    "Telugu",
    "Marathi",
)

EXPLANATION_STYLES: tuple[ExplanationStyle, ...] = (
    ExplanationStyle("Simple", "Use short sentences and everyday words."),
    ExplanationStyle("Step by step", "Explain each required action in order."),
    ExplanationStyle(
        "For an elderly family member",
        "Speak gently, avoid jargon, and repeat important dates and amounts.",
    ),
    ExplanationStyle("For a farmer", "Use practical examples and direct action points."),
    ExplanationStyle("For a student", "Explain unfamiliar terms and provide brief context."),
)


def style_labels() -> tuple[str, ...]:
    return tuple(style.label for style in EXPLANATION_STYLES)


def style_instruction(label: str) -> str:
    for style in EXPLANATION_STYLES:
        if style.label == label:
            return style.instruction
    raise ValueError(f"Unknown explanation style: {label}")
