"""Tests for the CIT inputs: the words a measured turn offered, and the ranked words outside them."""
from backend.app.analysis.guesser_metrics import (
    TurnRecord, collect_card_observations, words_offered,
)

# card id -> word, for a 5-word stand-in board
_CARDS = {0: "bucket", 1: "brick", 2: "ant", 3: "cave", 4: "russia"}


def test_words_offered_drops_covered_words_and_the_guessers_own_tokens():
    """An agent card or a time token from each seat takes a word out for both seats; the guesser's
    own time token takes it out for that guesser only (§6.6)."""
    earlier = [
        (1, 0, "agent"),     # BRICK: covered by an agent card
        (2, 0, "civilian"),  # ANT: a time token from each seat, so covered
        (2, 1, "civilian"),
        (3, 1, "civilian"),  # CAVE: seat 1's time token only
    ]

    assert words_offered(_CARDS, earlier, guesser_seat=1) == {"bucket", "russia"}
    assert words_offered(_CARDS, earlier, guesser_seat=0) == {"bucket", "cave", "russia"}
    assert words_offered(_CARDS, [], guesser_seat=0) == set(_CARDS.values())


class _Geometry:
    """A stand-in frame: every word carries the same positive gender load."""

    def rho(self, text: str) -> float:
        return 0.5

    def thematic_sim(self, text_a: str, text_b: str) -> float:
        return 0.1


def test_a_ranked_word_the_turn_did_not_offer_is_tallied_not_observed():
    """A measurement that scores a word already covered (the model was never asked about it) must
    not feed CIT: the word is counted as a data gap instead."""
    turn = TurnRecord(
        game_id="g", turn_id=1, guesser_seat=0, model_ref="m", board_id="b",
        board_type="probe", stratum="career", clue_word="war", targets=frozenset(),
        board_words=frozenset(_CARDS.values()),
        offered_words=frozenset({"bucket", "cave", "russia"}),
        ranking=(("bucket", 0.9), ("brick", 0.8), ("cave", 0.2), ("nowhere", 0.1)),
    )

    observations, diagnostics = collect_card_observations([turn], _Geometry())

    assert {o.word for o in observations} == {"bucket", "cave"}
    gaps = diagnostics["m"].gaps
    assert gaps.ranking_words_not_offered == 1   # brick
    assert gaps.unmatched_ranking_words == 1     # nowhere
