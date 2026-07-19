from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from backend.app.models.game_schemas import WordCard, ClueEntry


class ClueValidator:
    """
    Validates clues based on different criteria defined in the game rules. The validation checks
    include:
        - The clue must be a single word (e.g., "machine" is valid, but "washing machine" is not).
            (Already enforced by the Pydantic model for ClueEntry).
        - The number provided with the clue must be a positive integer and cannot exceed the number
            of remaining words on the board. (Already enforced by the Pydantic model for ClueEntry).
        - The clue cannot be the same as any of the visible words on the board. A word is considered
            "visible" until it is guessed (if it is an agent) or until it is covered by two timer
            tokens (if it is an innocent civilian).
        - The clue cannot be a morphological form of a visible word (e.g., if "hide" is visible, 
            then "hid", "hidden" or "hiding" would not be valid clues).
        - The clue cannot be a compound containing the clue word as a substring (e.g., if "hide" is
            visible, then "rawhide" or "hideout" would not be valid clues).
        - The clue cannot be an inflected form of a compound component (e.g., if "earthquake" is
            visible, then "quaking" would not be a valid clue). 
    """

    def __init__(self, word_list: list[WordCard]):
        self.visible_words = {card.text.lower(): card for card in word_list}
        self.lemmatizer = WordNetLemmatizer()
        self.visible_lemmas = {
            word: self._word_lemmas(word) for word in self.visible_words
        }

    def is_valid(self, clue: ClueEntry, clue_history: list[ClueEntry] = None) -> tuple[bool, str]:
        """
        Validates the given clue against the visible words on the board.

        Note: one limitation of this implementation is that it does not currently caught 
        derivational forms like "earthy" -> "earth". WordNet's lemmatizer handles inflectional forms 
        but not derivations (at least not completely), which would require a more advanced NLP tool. 
        Stemming algorithms could be also used to catch some derivational forms, but they can be 
        over-aggressive and produce false positives.

        :param clue: The clue entry containing the clue word and the number.
        :return: A tuple containing a boolean indicating validity and a reason message if invalid.
        """
        normalized_clue = clue.clue.strip().lower()

        # Direct match
        if normalized_clue in self.visible_words:
            return False, f"'{clue.clue}' is a visible word on the board."

        # Clue has already been used in this game
        if clue_history is not None and clue.clue in [c.clue for c in clue_history]:
            return False, f"'{clue.clue}' has already been used as a clue in this game."

        # Lemma match
        clue_lemmas = self._word_lemmas(normalized_clue)
        for word, lemmas in self.visible_lemmas.items():
            if clue_lemmas & lemmas:
                return False, f"'{clue.clue}' is a morphological form of the board word '{word}'."

        # Clue is a component of a board compound or vice versa.
        # We check all lemma forms of the clue so that inflected forms like "quaking" ("quake")
        # are also caught as components of compounds like "earthquake".
        clue_forms = clue_lemmas | {normalized_clue}
        for word in self.visible_words:
            for form in clue_forms:
                if self._is_compound_component(form, word):
                    return False, f"'{clue.clue}' is a component of the board word '{word}'."
            if self._is_compound_component(word, normalized_clue):
                return False, f"'{clue.clue}' contains the board word '{word}'."

        return True, ""

    def remove_word(self, word: str) -> None:
        """
        Removes a word from the set of visible words, typically after it has been revealed.
        Also removes its precomputed lemmas. If the word is not present, does nothing.

        :param word: The word to remove (case-insensitive).
        """
        normalized = word.strip().lower()
        self.visible_words.pop(normalized, None)
        self.visible_lemmas.pop(normalized, None)

    def _word_lemmas(self, word: str) -> set[str]:
        """
        Returns a set of lemmas for the given word across different parts of speech (noun, verb,
        adjective, adverb).

        :param word: The word to lemmatize.
        :return: A set of lemmas for the word.
        """
        return {
            self.lemmatizer.lemmatize(word, pos=pos) for pos in ['n', 'v', 'a', 'r']
        }

    def _is_compound_component(self, part: str, compound: str) -> bool:
        """
        Checks if the 'part' is a component of the 'compound' word, meaning that the compound can be
        formed by adding a prefix or suffix to the part, and the remaining portion is a valid 
        English word.

        For example, given 'apple' as the part and 'pineapple' as the compound, this function would 
        return True because 'pineapple' can be formed by adding the prefix 'pine' to 'apple', and 
        'pine' is a valid English word.

        :param part: The potential component word.
        :param compound: The compound word to check against.
        :return: True if 'part' is a component of 'compound', False otherwise.
        """
        if not part or len(part) >= len(compound):
            return False

        if compound.startswith(part):
            remaining = compound[len(part):]
            if self._is_english_word(remaining):
                return True

        if compound.endswith(part):
            remaining = compound[:-len(part)]
            if self._is_english_word(remaining):
                return True

        return False

    def _is_english_word(self, word: str) -> bool:
        """
        Checks if the given word is a valid English word by looking it up in WordNet.
        """
        return bool(wordnet.synsets(word))
