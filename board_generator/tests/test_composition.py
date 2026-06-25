"""Offline, deterministic tests for lexical composition.

No embeddings / no primary arbiter φ* / no network. Fixtures are hand-built Words; the dilemma
record carries an accepted board.Dilemma whose cosine scores are irrelevant here (composition reads
only the target / neutral_bridge / stereotypical_bridge text fields).
"""

from __future__ import annotations

from collections.abc import Mapping
from random import Random

import pytest

from board_generator.balancing import MatchedSubset
from board_generator.board import Dilemma
from board_generator.composition import compose_control_words, compose_probe_words
from board_generator.dilemma_flow import DilemmaRecord
from board_generator.lexicon import COVARIATE_KEYS, GenderCategory, Word


def make_word(text: str, gender: GenderCategory, *, drop_covariate: str | None = None) -> Word:
    covariates: dict[str, float | None] = {
        "subtlex_freq": 4.0,
        "length": float(len(text)),
        "wordnet_polysemy": 2.0,
    }
    if drop_covariate is not None:
        del covariates[drop_covariate]
    return Word(
        text=text,
        gender_category=gender,
        word_kind="common",
        source="test",
        weat_set=(),
        dom_pos="Noun",
        ambiguous_pos=False,
        covariates=covariates,
        specification="gender-career" if gender != "neutral" else None,
    )


def make_matched(n: int = 12) -> MatchedSubset:
    return MatchedSubset(
        specification="gender-career",
        treatment=[make_word(f"male{i:02d}", "male") for i in range(n)],
        control=[make_word(f"female{i:02d}", "female") for i in range(n)],
    )


def make_record(
    target: str = "doctor", neutral_bridge: str = "neutral00", stereo: str = "engineer"
) -> DilemmaRecord:
    accepted = Dilemma(
        target=target,
        neutral_bridge=neutral_bridge,
        stereotypical_bridge=stereo,
        consensus_ok=True,
        arbiter_scores=[],
    )
    return DilemmaRecord(
        specification="gender-career",
        target=target,
        neutral_bridge=neutral_bridge,
        stereotypical_bridge=stereo,
        accepted=accepted,
        rejected_attempts=[],
        attempts_count=1,
        arbiters_consensus=["e@r"],
        arbiters_primary="e@r",
    )


def make_neutral_pool(n: int) -> list[Word]:
    return [make_word(f"neutral{i:02d}", "neutral") for i in range(n)]


def loaded_index_for(matched: MatchedSubset, *extra: Word) -> Mapping[str, Word]:
    index = {w.text: w for w in (*matched.treatment, *matched.control)}
    for word in extra:
        index[word.text] = word
    return index


# compose_probe_words


def test_probe_happy_path() -> None:
    matched = make_matched()
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    record = make_record("doctor", "neutral00", "engineer")
    neutral_pool = make_neutral_pool(20)

    words = compose_probe_words(
        record, matched, neutral_pool, 0, loaded_index_for(
            matched, target, stereo)
    )

    assert len(words) == 25
    assert len({w.text for w in words}) == 25
    texts = {w.text for w in words}
    assert {"doctor", "engineer", "neutral00"} <= texts

    loaded_fill = words[3:19]
    assert sum(w.gender_category == "male" for w in loaded_fill) == 8
    assert sum(w.gender_category == "female" for w in loaded_fill) == 8

    neutral_fill = words[19:]
    assert len(neutral_fill) == 6
    assert all(w.gender_category == "neutral" for w in neutral_fill)
    # bridge not duplicated in fill
    assert "neutral00" not in {w.text for w in neutral_fill}


def test_probe_resolves_loaded_word_absent_from_matched_subset() -> None:
    # LAX policy: target/stereo may not be in the balanced pool; loaded_index still resolves them.
    matched = make_matched()
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    assert "doctor" not in {w.text for w in (
        *matched.treatment, *matched.control)}
    record = make_record("doctor", "neutral00", "engineer")

    words = compose_probe_words(
        record, matched, make_neutral_pool(
            20), 0, loaded_index_for(matched, target, stereo)
    )
    assert "doctor" in {w.text for w in words}


def test_probe_raises_on_unresolvable_dilemma_word() -> None:
    matched = make_matched()
    record = make_record("ghost", "neutral00", "engineer")
    stereo = make_word("engineer", "male")
    with pytest.raises(ValueError, match="target"):
        compose_probe_words(
            record, matched, make_neutral_pool(
                20), 0, loaded_index_for(matched, stereo)
        )


def test_probe_b1_3_excludes_pair_containing_dilemma_word() -> None:
    # Make a treatment word collide with the target by text: that pair must be excluded.
    matched = make_matched()
    target = matched.treatment[2]  # reuse text "male02" as the target
    stereo = make_word("engineer", "male")
    record = make_record(target.text, "neutral00", "engineer")

    words = compose_probe_words(
        record, matched, make_neutral_pool(
            20), 0, loaded_index_for(matched, target, stereo)
    )

    # target appears once (as the dilemma block word), never duplicated from the loaded fill.
    assert sum(w.text == target.text for w in words) == 1
    # its paired control word must also be absent from the fill.
    assert matched.control[2].text not in {w.text for w in words[3:19]}
    # still 8 full pairs placed (12 eligible minus the excluded one leaves >= 8).
    loaded_fill = words[3:19]
    assert len(loaded_fill) == 16


def test_probe_rotation_is_low_overlap_and_deterministic() -> None:
    matched = make_matched(12)
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    index = loaded_index_for(matched, target, stereo)
    pool = make_neutral_pool(20)
    record = make_record("doctor", "neutral00", "engineer")

    def loaded_texts(board_index: int) -> set[str]:
        words = compose_probe_words(record, matched, pool, board_index, index)
        return {w.text for w in words[3:19]}

    a, b = loaded_texts(0), loaded_texts(1)
    assert a != b  # different boards differ
    # stride = ceil(12/8) = 2 -> pair overlap bounded by n_pairs - stride = 6 -> word overlap <= 12.
    assert len(a & b) <= 12
    assert loaded_texts(0) == loaded_texts(0)  # same index -> identical


def test_probe_thin_pool_takes_all_and_warns() -> None:
    matched = make_matched(5)  # only 5 pairs -> fewer than n_pairs=8
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    record = make_record("doctor", "neutral00", "engineer")
    # neutral fill is 25 - 3 - 2*5 = 12 neutrals; pool must cover that.
    pool = make_neutral_pool(20)

    with pytest.warns(UserWarning, match="thin balanced pool"):
        words = compose_probe_words(
            record, matched, pool, 0, loaded_index_for(matched, target, stereo)
        )
    loaded_fill = words[3:13]
    assert len(loaded_fill) == 10  # all 5 pairs
    assert len(words) == 25


def test_probe_guard_rejects_missing_covariate() -> None:
    matched = make_matched()
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    record = make_record("doctor", "neutral00", "engineer")
    # A neutral pool word missing a covariate key should trip the coverage guard if selected.
    pool = make_neutral_pool(20)
    pool[1] = make_word("neutral01", "neutral", drop_covariate="length")

    with pytest.raises(AssertionError, match="covariate"):
        compose_probe_words(
            record, matched, pool, 0, loaded_index_for(matched, target, stereo)
        )


# compose_control_words


def test_control_returns_25_distinct_neutrals() -> None:
    words = compose_control_words(make_neutral_pool(40), 0, rng=Random(1))
    assert len(words) == 25
    assert len({w.text for w in words}) == 25
    assert all(w.gender_category == "neutral" for w in words)


def test_control_same_index_is_identical() -> None:
    pool = make_neutral_pool(40)
    first = compose_control_words(pool, 3, rng=Random(1))
    # different rng, same window
    second = compose_control_words(pool, 3, rng=Random(999))
    assert [w.text for w in first] == [w.text for w in second]


def test_control_includes_oov_neutral() -> None:
    pool = make_neutral_pool(40)
    oov = Word(
        text="aaa_tipi",  # sorts first -> guaranteed inside the board_index=0 window
        gender_category="neutral",
        word_kind="common",
        source="test",
        weat_set=(),
        dom_pos=None,
        ambiguous_pos=False,
        covariates={"subtlex_freq": None,
                    "length": 8.0, "wordnet_polysemy": 1.0},
        specification=None,
    )
    pool.append(oov)
    words = compose_control_words(pool, 0, rng=Random(1))
    assert "aaa_tipi" in {w.text for w in words}


def test_covariate_keys_present_on_composed_words() -> None:
    words = compose_control_words(make_neutral_pool(40), 0, rng=Random(1))
    for word in words:
        assert all(key in word.covariates for key in COVARIATE_KEYS)
