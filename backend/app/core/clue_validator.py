import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from backend.app.models.game_schemas import WordCard, ClueEntry

MORPHOLEX_PATH = Path(__file__).resolve().parents[3] / "data" / "morpholex" / "morpholex_en.tsv"

# Prefixes in MorphoLex that are words of their own, so the words they start are compounds for the
# rules: "outside" is out + side, and "upset" is up + set.
_PARTICLES = frozenset({
    "after", "back", "by", "cross", "down", "half", "in", "off", "on", "out", "over", "through",
    "under", "up",
})


class _Unit(NamedTuple):
    """A root group of a MorphoLex segmentation, or a prefix that is a word of its own."""
    stem: str  # the group's content, fused affixes included: "(medic)>al>", "<trans<(port)"
    roots: tuple[str, ...]  # the roots in it: ("medic",), ("port",)
    fused_prefix: bool  # whether a prefix is fused into it: True for "<trans<(port)"


class _Morphology:
    """
    Word segmentations from MorphoLex-en (see data/morpholex/README.md for the source, the notation
    and the license).

    A segmentation is made of root groups in braces, with prefixes and suffixes. MorphoLex puts
    inside the braces the affixes that are fused with the root, and outside those that make a
    transparent derivation: "potent" is {(pot)>ant>} and "potter" is {(pot)}>er>, "transport" is
    {<trans<(port)} and "unstable" is <un<{(stable)}. A compound has a group per part:
    "earthquake" is {(earth)}{(quake)}.
    """

    def __init__(self, path: Path):
        self._segmentations: dict[str, str] = {}
        with open(path, encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                word, segmentation = line.rstrip("\r\n").split("\t")
                self._segmentations[word] = segmentation
        self._units: dict[str, tuple[_Unit, ...] | None] = {}

    def __contains__(self, word: str) -> bool:
        return word in self._segmentations

    def units(self, word: str) -> tuple[_Unit, ...] | None:
        """
        Returns the units of a word: one per root group, plus one per prefix that is a word of its
        own (_PARTICLES). Transparent affixes are left out, so "earthy" has the same unit as
        "earth", and "upstairs" the units of "up" and "stairs".

        :param word: The lowercase word.
        :return: The units, or None if MorphoLex does not have the word.
        """
        if word not in self._units:
            segmentation = self._segmentations.get(word)
            self._units[word] = None if segmentation is None else self._parse(word, segmentation)
        return self._units[word]

    def _parse(self, word: str, segmentation: str) -> tuple[_Unit, ...] | None:
        units = []
        i = 0
        while i < len(segmentation):
            char = segmentation[i]
            if char == "<":  # a prefix outside the root groups
                end = self._closing(segmentation, "<", i)
                prefix = segmentation[i + 1:end]
                if prefix in _PARTICLES:
                    units.append(_Unit(f"({prefix})", (prefix,), False))
            elif char == "{":  # a root group, which can hold braces of its own
                end, depth = i, 0
                while end < len(segmentation):
                    depth += {"{": 1, "}": -1}.get(segmentation[end], 0)
                    if depth == 0:
                        break
                    end += 1
                units += self._group_units(word, segmentation[i + 1:end])
            elif char == "(":  # a root outside the braces: "blueberry" is {(blue)}(berry)
                end = self._closing(segmentation, ")", i)
                root = segmentation[i + 1:end]
                units.append(_Unit(f"({root})", (root,), False))
            elif char == ">":  # a suffix outside the root groups
                end = self._closing(segmentation, ">", i)
            else:
                end = i
            i = end + 1
        return tuple(units) or None

    def _group_units(self, word: str, group: str) -> list[_Unit]:
        roots = tuple(re.findall(r"\(([^()<>{}]+)\)", group))
        if not roots:
            return []
        # Some compounds are a single group with several roots, "christmas" is {(christ)(mas)} and
        # "scapegoat" is {(scape)(goat)}. Without fused affixes, and with a root that is a word
        # spelled out in the compound, the roots are parts of their own. "geometry" stays a group:
        # it has a fused suffix, {(geo)(meter)>y>}.
        fused_affixes = re.sub(r"\([^()<>{}]+\)", "", group)
        if len(roots) > 1 and not fused_affixes and any(
                len(root) >= 3 and root in word and root in self for root in roots):
            return [_Unit(f"({root})", (root,), False) for root in roots]
        return [_Unit(group, roots, "<" in group)]

    @staticmethod
    def _closing(segmentation: str, char: str, start: int) -> int:
        end = segmentation.find(char, start + 1)
        return len(segmentation) - 1 if end < 0 else end


@lru_cache(maxsize=1)
def _morphology() -> _Morphology:
    return _Morphology(MORPHOLEX_PATH)


def _stems(units: tuple[_Unit, ...]) -> Counter:
    return Counter(unit.stem for unit in units)


def _same_family(a: tuple[_Unit, ...], b: tuple[_Unit, ...], morphology: _Morphology) -> bool:
    """
    Whether two words are forms of each other: they have the same units, fused affixes included
    ("potter" and "pot"; "unstable" and "stable"). MorphoLex also fuses some transparent suffixes
    ("medical" is {(medic)>al>}), so a word that only adds fused suffixes to a root that is a word
    is a form of that word too ("medical" and "medic").
    """
    if _stems(a) == _stems(b):
        return True
    if len(a) == 1 and len(b) == 1:
        x, y = a[0], b[0]
        if (x.roots == y.roots and len(x.roots) == 1 and x.roots[0] in morphology
                and not x.fused_prefix and not y.fused_prefix):
            return f"({x.roots[0]})" in (x.stem, y.stem)
    return False


def _is_part(part: tuple[_Unit, ...], compound: tuple[_Unit, ...]) -> bool:
    """Whether the units of 'part' are some, but not all, of the units of the compound."""
    part_stems, compound_stems = _stems(part), _stems(compound)
    return len(compound) >= 2 and part_stems != compound_stems and not part_stems - compound_stems


def _root_unit(word: str) -> tuple[_Unit, ...]:
    """The units of a word taken as a bare root, for a word MorphoLex does not have."""
    return (_Unit(f"({word})", (word,), False),)


def _shares_spelling(a: str, b: str) -> bool:
    """Whether a and b share the spelling of the shorter one, except perhaps its last letter."""
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    return common >= max(3, min(len(a), len(b)) - 1)


class ClueValidator:
    """
    Validates clues based on different criteria defined in the game rules. The validation checks
    include:
        - The clue must be a single word (e.g., "machine" is valid, but "washing machine" is not),
            and so must a proper name ("Michelangelo" is valid, "Leonardo da Vinci" is not).
        - The clue cannot be the same as any of the visible words on the board (§8.1.4). A word is
            considered "visible" until it is guessed (if it is an agent) or until it is covered by
            two timer tokens (if it is an innocent civilian). Case, surrounding punctuation and a
            possessive "'s" are ignored, so "Crown's" is the visible word "crown".
        - The clue cannot be a form of a visible word (§8.1.5): an inflected form (if "hide" is
            visible, then "hid", "hidden" or "hiding"), a derived form ("kingdom" with "king"
            visible, and "king" with "kingdom"), or a compound that contains the visible word
            ("rawhide" with "hide").
        - The clue cannot be a part of a visible compound, or a form of a part (§8.1.6): with
            "earthquake" visible, "earth", "quake", "earthy" and "quaking" are not valid.

    The rules are applied as written, with nothing added. A word that only shares letters with a
    visible word is valid ("hideous" with "hide", "ear" and "hearth" with "earthquake"), and so is
    a word that only shares a part with it: "raincoat" with "rainbow" visible (two compounds with a
    common part), and "firearm" with "armament" (a compound with the root of a derived word).

    Inflected forms come from WordNet's lemmatizer. Derived forms and compound parts come from the
    word segmentations of MorphoLex-en (see _Morphology), with WordNet's derivational links for the
    derivations MorphoLex takes for words of their own ("marriage", "patience"). For a word
    MorphoLex does not have, the validator falls back to WordNet's derivational links and to
    splitting the word in two pieces of at least three letters.

    Repeating a clue already given in the game is allowed, as in the Duet rules (§8.1), so the
    validator takes no clue history.

    The count is not checked here. ClueEntry requires it to be at least 1 (the platform does not
    support zero clues, see "Deviations from the Duet rules" in the README), and the rules set no
    maximum (§14).

    Known limitations, accepted on purpose (see "Deviations from the Duet rules" in the README):
        - MorphoLex segments some words by their etymology, so a few legal clues are flagged:
            "tenant" with "ten" visible, "potent" with "potter", "irony" with "ironing", "dent"
            with "dentist", "cookie" with "cooking".
        - MorphoLex takes some compounds and derivations for words of their own, and those pass
            when WordNet does not link them either: "wolf" with "werewolf" visible, "asleep" with
            "sleep", "repay" with "pay".
        - About 2% of the board words and clues are not in MorphoLex. For those, the fallback
            catches less: it misses "-ly" adverbs ("happily" with "happy") and compound pieces of
            one or two letters, and it takes WordNet's derivational links as they are, so a link
            hidden by the spelling ("flight" with "flee") is caught too.
    """

    def __init__(self, word_list: list[WordCard]):
        self.visible_words = {card.text.lower(): card for card in word_list}
        self.lemmatizer = WordNetLemmatizer()
        self.morphology = _morphology()
        self.visible_lemmas = {
            word: self._word_lemmas(word) for word in self.visible_words
        }
        self.visible_analyses = {
            word: self._analyses(word) for word in self.visible_words
        }

    def is_valid(self, clue: ClueEntry) -> tuple[bool, str]:
        """
        Validates the given clue against the visible words on the board.

        :param clue: The clue entry containing the clue word and the number.
        :return: A tuple containing a boolean indicating validity and a reason message if invalid.
        """
        normalized_clue = self._normalize(clue.clue)

        # A clue is one word (§6.2), and a proper name of several words is not allowed (§8.1.10)
        if len(normalized_clue.split()) > 1:
            return False, f"'{clue.clue}' is not a single word."

        # Direct match
        if normalized_clue in self.visible_words:
            return False, f"'{clue.clue}' is a visible word on the board."

        # Lemma match
        clue_lemmas = self._word_lemmas(normalized_clue)
        for word, lemmas in self.visible_lemmas.items():
            if clue_lemmas & lemmas:
                return False, f"'{clue.clue}' is a morphological form of the board word '{word}'."

        # Derived forms and compounds (§8.1.5-6), with MorphoLex if it has both words.
        clue_forms = clue_lemmas | {normalized_clue}
        clue_analyses = self._analyses(normalized_clue)
        links = self._derivational_links(clue_forms)
        for word in self.visible_words:
            if clue_analyses and self.visible_analyses[word]:
                reason = self._morpholex_reason(clue, clue_forms, clue_analyses, links, word)
            else:
                reason = self._fallback_reason(clue, normalized_clue, clue_forms, clue_analyses,
                                               links, word)
            if reason:
                return False, reason

        return True, ""

    def remove_word(self, word: str) -> None:
        """
        Removes a word from the set of visible words, typically after it has been revealed.
        Also removes its precomputed lemmas and analyses. If the word is not present, does nothing.

        :param word: The word to remove (case-insensitive).
        """
        normalized = word.strip().lower()
        self.visible_words.pop(normalized, None)
        self.visible_lemmas.pop(normalized, None)
        self.visible_analyses.pop(normalized, None)

    def _morpholex_reason(self, clue: ClueEntry, clue_forms: set[str],
                          clue_analyses: list[tuple[_Unit, ...]], links: set[tuple[str, str]],
                          word: str) -> str | None:
        """
        Checks the clue against a visible word when MorphoLex has both.

        :return: The reason the clue is invalid, or None if it is valid against this word.
        """
        for clue_units in clue_analyses:
            for word_units in self.visible_analyses[word]:
                if _same_family(clue_units, word_units, self.morphology):
                    return f"'{clue.clue}' is a derived form of the board word '{word}'."
                if _is_part(clue_units, word_units):
                    return self._component_reason(clue, clue_forms, clue_units, word)
                # A compound contains the board word only if it spells it out: "rawhide" contains
                # "hide", but "firearm" does not contain "armament", only its root.
                if _is_part(word_units, clue_units) and self._spells_out(clue_forms, word):
                    return f"'{clue.clue}' contains the board word '{word}'."

        # Derivations that MorphoLex takes for words of their own: "marriage" is {(marriage)}, and
        # WordNet links it to "marry". Only links that keep the spelling are taken, which leaves
        # out the ones that are not a form in the rules' sense ("flight" and "flee").
        if any(derived in self.visible_lemmas[word] and _shares_spelling(form, derived)
               for form, derived in links):
            return f"'{clue.clue}' is a derived form of the board word '{word}'."
        return None

    def _fallback_reason(self, clue: ClueEntry, normalized_clue: str, clue_forms: set[str],
                         clue_analyses: list[tuple[_Unit, ...]], links: set[tuple[str, str]],
                         word: str) -> str | None:
        """
        Checks the clue against a visible word when MorphoLex lacks one of them, or both. It uses
        MorphoLex for the word it has, and WordNet for the rest.

        :return: The reason the clue is invalid, or None if it is valid against this word.
        """
        derived_forms = {derived for _, derived in links} - clue_forms
        if derived_forms & self.visible_lemmas[word]:
            return f"'{clue.clue}' is a derived form of the board word '{word}'."

        # The clue, or a derived form of it, is a part of the board compound.
        word_analyses = self.visible_analyses[word]
        for forms, derived in ((clue_forms, False), (derived_forms, True)):
            for form in forms:
                if word_analyses:
                    is_part = any(_is_part(_root_unit(form), units) for units in word_analyses)
                else:
                    is_part = self._is_compound_component(form, word)
                if is_part and derived:
                    return (f"'{clue.clue}' is a derived form of '{form}', a component of the "
                            f"board word '{word}'.")
                if is_part:
                    return f"'{clue.clue}' is a component of the board word '{word}'."

        # The clue is a compound that contains the board word.
        if clue_analyses:
            contains = self._spells_out(clue_forms, word) and any(
                _is_part(_root_unit(form), units)
                for form in self.visible_lemmas[word] | {word} for units in clue_analyses)
        else:
            contains = self._is_compound_component(word, normalized_clue)
        if contains:
            return f"'{clue.clue}' contains the board word '{word}'."
        return None

    def _component_reason(self, clue: ClueEntry, clue_forms: set[str],
                          clue_units: tuple[_Unit, ...], word: str) -> str:
        """The reason for a clue that is a part of a visible compound, or a form of a part."""
        if len(clue_units) == 1 and clue_units[0].stem not in {f"({form})" for form in clue_forms}:
            part = "".join(clue_units[0].roots)
            return (f"'{clue.clue}' is a derived form of '{part}', a component of the board word "
                    f"'{word}'.")
        return f"'{clue.clue}' is a component of the board word '{word}'."

    def _spells_out(self, clue_forms: set[str], word: str) -> bool:
        """
        Whether the clue, or one of its lemmas, spells out the board word or one of its lemmas:
        "withdraw" and "withdrew" (lemma "withdraw") spell out "drawing" (lemma "draw").
        """
        word_forms = self.visible_lemmas[word] | {word}
        return any(word_form in clue_form for clue_form in clue_forms for word_form in word_forms)

    @staticmethod
    def _normalize(clue: str) -> str:
        """
        Lowercases the clue and strips the punctuation around it and a possessive "'s", so that
        "CROWN!", "crown." and "Crown's" are all "crown".
        """
        normalized = clue.strip().lower().replace("’", "'")
        normalized = re.sub(r"^\W+|\W+$", "", normalized)
        return re.sub(r"'s$", "", normalized)

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

    def _analyses(self, word: str) -> list[tuple[_Unit, ...]]:
        """
        Returns the MorphoLex units of the word and of its lemmas, without repeats. A hyphenated
        word is a compound of its parts: "rain-soaked" has the units of "rain" and of "soaked".

        :param word: The lowercase word.
        :return: The analyses, empty if MorphoLex has neither the word nor its lemmas.
        """
        if "-" in word:
            parts = [self._analyses(part) for part in word.split("-") if part]
            if not parts or not all(parts):
                return []
            return [tuple(unit for analyses in parts for unit in analyses[0])]
        analyses = []
        for form in [word] + sorted(self._word_lemmas(word) - {word}):
            units = self.morphology.units(form)
            if units and units not in analyses:
                analyses.append(units)
        return analyses

    def _derivational_links(self, words: set[str]) -> set[tuple[str, str]]:
        """
        Returns the derivationally related forms of the given words in WordNet, across all their
        senses and parts of speech (e.g., "earthy" -> "earth", "king" -> "kingdom", "kingly").

        :param words: The words (lemmas) to look up.
        :return: A set of (word, lowercase derived form) pairs. Multi-word forms keep WordNet's
            underscores.
        """
        return {
            (word, related.name().lower())
            for word in words
            for lemma in wordnet.lemmas(word)
            for related in lemma.derivationally_related_forms()
        }

    def _is_compound_component(self, part: str, compound: str) -> bool:
        """
        Checks if the 'part' is a component of the 'compound' word, meaning that the compound can be
        formed by adding a prefix or suffix to the part, and the remaining portion is a valid
        English word. Only used for words MorphoLex does not have.

        For example, given 'apple' as the part and 'pineapple' as the compound, this function would
        return True because 'pineapple' can be formed by adding the prefix 'pine' to 'apple', and
        'pine' is a valid English word.

        Both pieces must have at least three letters, or the remaining one must be a word like "up"
        ("sunup"): WordNet has entries for letters, symbols and abbreviations, so shorter pieces
        would make 'river' a component of 'driver' ("d").

        :param part: The potential component word.
        :param compound: The compound word to check against.
        :return: True if 'part' is a component of 'compound', False otherwise.
        """
        if len(part) < 3 or len(part) >= len(compound):
            return False

        remainders = []
        if compound.startswith(part):
            remainders.append(compound[len(part):])
        if compound.endswith(part):
            remainders.append(compound[:-len(part)])
        return any(
            (len(remaining) >= 3 or remaining in _PARTICLES) and self._is_english_word(remaining)
            for remaining in remainders
        )

    def _is_english_word(self, word: str) -> bool:
        """
        Checks if the given word is a valid English word by looking it up in WordNet.
        """
        return bool(wordnet.synsets(word))
