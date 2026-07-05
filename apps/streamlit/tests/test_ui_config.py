import pytest

from ui_config import LANGUAGES, TRANSLATIONS, style_instruction, style_labels, text


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


def test_all_languages_have_the_same_translation_keys() -> None:
    expected = set(TRANSLATIONS["English"])

    assert all(set(catalog) == expected for catalog in TRANSLATIONS.values())


@pytest.mark.parametrize("language", LANGUAGES)
def test_essential_ui_text_exists_in_each_language(language: str) -> None:
    assert text(language, "title")
    assert text(language, "upload_label")
    assert text(language, "chat_placeholder")
    assert len(style_labels(language)) == 5
