from nltk.stem import WordNetLemmatizer
from backend.app.models.game_schemas import WordCard, ClueEntry


class ClueValidator:
    """
    Validates clues based on different criteria defined in the game rules. The validation checks
    include:
        - The clue must be a single word (e.g., "clean" is valid, but "washing machine" is not). 
        - The number provided with the clue must be a positive integer and cannot exceed the number
            of remaining words on the board.
        - The clue cannot be the same as any of the visible words on the board. A word is considered
            "visible" until it is guessed (if it is an agent) or until it is covered by two timer
            tokens (if it is an innocent civilian).
        - The clue cannot be a morphological form of a visible word (e.g., if "hide" is visible, 
            then "hid", "hidden" or "hiding" would not be valid clues).
        - The clue cannot be a compound containing the clue word as a substring (e.g., if "hide" is
            visible, then "rawhide" or "hideout" would not be valid clues).
        - The clue cannot be an inflected form of a compound component (e.g., if "earthquake" is
            visible, then "earthy" or "quaking" would not be valid clues).
    """

    def __init__(self, word_list: set[WordCard]):
        pass

    def is_valid(self, clue: ClueEntry) -> tuple[bool, str]:
        """
        Validates the given clue against the visible words on the board.

        :param clue: The clue entry containing the clue word and the number.
        :return: A tuple containing a boolean indicating validity and a reason message if invalid. 
        """
        # Single word
        # Direct match
        # Lemma match
        # Clue is a component of a board compound
        # Board word is a component of the clue compound
        pass
