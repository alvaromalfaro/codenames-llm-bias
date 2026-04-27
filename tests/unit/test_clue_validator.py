import pytest
from backend.app.core.clue_validator import ClueValidator
from backend.app.models.game_schemas import CardRole, ClueEntry, WordCard


def _make_card(id: int, text: str) -> WordCard:
    return WordCard(
        id=id,
        text=text,
        llm_perspective_role=CardRole.CIVILIAN,
        human_perspective_role=CardRole.CIVILIAN,
    )


def _make_clue(word: str) -> ClueEntry:
    return ClueEntry(clue=word, count=1, clue_giver=0)


@pytest.fixture
def validator_hide() -> ClueValidator:
    return ClueValidator([_make_card(0, "HIDE")])


@pytest.fixture
def validator_earthquake() -> ClueValidator:
    return ClueValidator([_make_card(0, "EARTHQUAKE")])


def test_direct_match_exact(validator_hide):
    """A clue that exactly matches a visible board word is invalid."""
    valid, reason = validator_hide.is_valid(_make_clue("hide"))

    assert not valid
    assert "visible word" in reason


def test_direct_match_different_casing(validator_hide):
    """The direct match check is case-insensitive."""
    valid, reason = validator_hide.is_valid(_make_clue("Hide"))

    assert not valid
    assert "visible word" in reason


def test_lemma_match_inflected_verb(validator_hide):
    """An inflected verb form of a visible word is invalid ("hiding" → "hide")."""
    valid, reason = validator_hide.is_valid(_make_clue("hiding"))

    assert not valid
    assert "morphological form" in reason


def test_lemma_match_past_tense(validator_hide):
    """The past tense of a visible word is invalid ("hid" → "hide")."""
    valid, reason = validator_hide.is_valid(_make_clue("hid"))

    assert not valid
    assert "morphological form" in reason


def test_lemma_match_unrelated_word_same_ending(validator_hide):
    """
    A word that shares letters with a board word but has a different root is valid.
    "ride" ends in "-ide" like "hide" but lemmatizes to "ride", not "hide".
    """
    valid, _ = validator_hide.is_valid(_make_clue("ride"))

    assert valid


def test_compound_prefix_component(validator_earthquake):
    """
    A word that forms the prefix of a compound board word is invalid.
    "earth" + "quake" = "earthquake", and "quake" is a valid English word.
    """
    valid, reason = validator_earthquake.is_valid(_make_clue("earth"))

    assert not valid
    assert "component" in reason


def test_compound_suffix_component(validator_earthquake):
    """
    A word that forms the suffix of a compound board word is invalid.
    "earth" + "quake" = "earthquake", and "earth" is a valid English word.
    """
    valid, reason = validator_earthquake.is_valid(_make_clue("quake"))

    assert not valid
    assert "component" in reason


def test_compound_clue_contains_board_word():
    """
    A clue that is itself a compound containing a board word is invalid.
    "raw" + "hide" = "rawhide", and "raw" is a valid English word.
    """
    validator = ClueValidator([_make_card(0, "HIDE")])
    valid, reason = validator.is_valid(_make_clue("rawhide"))

    assert not valid
    assert "contains" in reason


def test_compound_shared_letters_not_a_component(validator_earthquake):
    """
    A word that is a substring of a board word but NOT a compound component is valid.
    "ear" appears inside "earthquake", but removing it leaves "thquake" which is not
    a valid English word, so "ear" is not a compound component.
    """
    valid, _ = validator_earthquake.is_valid(_make_clue("ear"))

    assert valid


def test_compound_inflected_form_of_component(validator_earthquake):
    """
    An inflected form of a compound component is invalid.
    "quaking" lemmatizes to "quake", which is a suffix component of "earthquake".
    """
    valid, reason = validator_earthquake.is_valid(_make_clue("quaking"))

    assert not valid
    assert "component" in reason


def test_known_limitation_derivational_form_not_caught():
    """
    Derivational adjective forms of board words are NOT caught by this implementation.
    "earthy" is derived from "earth" via the adjectival suffix "-y", but
    WordNetLemmatizer only handles inflectional morphology (tense, number, etc.),
    not derivational morphology. Fixing this requires a dedicated derivational
    morphology tool or lexicon.
    """
    validator = ClueValidator([_make_card(0, "EARTHQUAKE")])
    valid, _ = validator.is_valid(_make_clue("earthy"))

    assert valid  # incorrectly treated as valid — known limitation


def test_valid_unrelated_clue(validator_hide):
    """A clue completely unrelated to any board word is valid."""
    valid, reason = validator_hide.is_valid(_make_clue("ocean"))

    assert valid
    assert reason == ""


def test_remove_word_allows_direct_match(validator_hide):
    """After removing a word from the visible words, its exact form is no longer invalid."""
    valid, _ = validator_hide.is_valid(_make_clue("hide"))
    assert not valid

    validator_hide.remove_word("HIDE")

    valid, _ = validator_hide.is_valid(_make_clue("hide"))
    assert valid


def test_remove_word_allows_morphological_forms(validator_hide):
    """After removing a word, its morphological forms are also no longer invalid."""
    valid, _ = validator_hide.is_valid(_make_clue("hiding"))
    assert not valid

    validator_hide.remove_word("HIDE")

    valid, _ = validator_hide.is_valid(_make_clue("hiding"))
    assert valid


def test_remove_word_is_case_insensitive(validator_hide):
    """remove_word normalizes casing, so "HIDE", "Hide", and "hide" all remove the same entry."""
    validator_hide.remove_word("Hide")

    valid, _ = validator_hide.is_valid(_make_clue("hide"))
    assert valid


def test_remove_nonexistent_word_does_not_raise(validator_hide):
    """Removing a word that is not on the board does not raise an error."""
    validator_hide.remove_word("NOTAWORD")
