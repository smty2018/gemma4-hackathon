import pytest

from ui_config import LANGUAGES, style_instruction, style_labels


def test_language_options_cover_initial_mvp() -> None:
    assert {"English", "Hindi", "Bengali"}.issubset(LANGUAGES)


def test_explanation_styles_are_unique() -> None:
    labels = style_labels()
    assert len(labels) == len(set(labels))
    assert "Simple" in labels
    assert "For an elderly family member" in labels


def test_unknown_style_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown explanation style"):
        style_instruction("Unknown")
