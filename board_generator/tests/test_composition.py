"""Offline, deterministic tests for lexical composition.

No embeddings / no primary arbiter φ* / no network. Fixtures are hand-built Words; the dilemma
record carries an accepted board.Dilemma whose cosine scores are irrelevant here (composition reads
only the target / neutral_bridge / stereotypical_bridge text fields).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from random import Random

import pytest

from board_generator.balancing import MatchedSubset
from board_generator.board import Dilemma
from board_generator.composition import (
    compose_control_words,
    compose_probe_words,
    pair_selection_overlap,
)
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
            matched, target, stereo), rng=Random(0)
    )

    assert len(words) == 25
    assert len({w.text for w in words}) == 25
    texts = {w.text for w in words}
    assert {"doctor", "engineer", "neutral00"} <= texts

    loaded_fill = words[3:15]
    assert sum(w.gender_category == "male" for w in loaded_fill) == 6
    assert sum(w.gender_category == "female" for w in loaded_fill) == 6

    neutral_fill = words[15:]
    assert len(neutral_fill) == 10
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
            20), 0, loaded_index_for(matched, target, stereo), rng=Random(0)
    )
    assert "doctor" in {w.text for w in words}


def test_probe_raises_on_unresolvable_dilemma_word() -> None:
    matched = make_matched()
    record = make_record("ghost", "neutral00", "engineer")
    stereo = make_word("engineer", "male")
    with pytest.raises(ValueError, match="target"):
        compose_probe_words(
            record, matched, make_neutral_pool(
                20), 0, loaded_index_for(matched, stereo), rng=Random(0)
        )


def test_probe_b1_3_excludes_pair_containing_dilemma_word() -> None:
    # Make a treatment word collide with the target by text: that pair must be excluded.
    matched = make_matched()
    target = matched.treatment[2]  # reuse text "male02" as the target
    stereo = make_word("engineer", "male")
    record = make_record(target.text, "neutral00", "engineer")

    words = compose_probe_words(
        record, matched, make_neutral_pool(
            20), 0, loaded_index_for(matched, target, stereo), rng=Random(0)
    )

    # target appears once (as the dilemma block word), never duplicated from the loaded fill.
    assert sum(w.text == target.text for w in words) == 1
    # its paired control word must also be absent from the fill.
    assert matched.control[2].text not in {w.text for w in words[3:15]}
    # still 6 full pairs placed (12 eligible minus the excluded one leaves >= 6).
    loaded_fill = words[3:15]
    assert len(loaded_fill) == 12


def test_probe_selection_is_deterministic_and_index_dependent() -> None:
    matched = make_matched(12)
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    index = loaded_index_for(matched, target, stereo)
    pool = make_neutral_pool(20)
    record = make_record("doctor", "neutral00", "engineer")

    def loaded_texts(board_index: int) -> set[str]:
        words = compose_probe_words(
            record, matched, pool, board_index, index, rng=Random(0))
        return {w.text for w in words[3:15]}

    # same index -> identical pick
    assert loaded_texts(0) == loaded_texts(0)
    # different boards generally differ
    assert loaded_texts(0) != loaded_texts(1)


def test_probe_thin_pool_takes_all_and_warns() -> None:
    matched = make_matched(5)  # only 5 pairs -> fewer than n_pairs=6
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    record = make_record("doctor", "neutral00", "engineer")
    # neutral fill is 25 - 3 - 2*5 = 12 neutrals; pool must cover that.
    pool = make_neutral_pool(20)

    with pytest.warns(UserWarning, match="thin balanced pool"):
        words = compose_probe_words(
            record, matched, pool, 0, loaded_index_for(
                matched, target, stereo), rng=Random(0)
        )
    loaded_fill = words[3:13]
    assert len(loaded_fill) == 10  # all 5 pairs
    assert len(words) == 25


def test_probe_guard_rejects_missing_covariate() -> None:
    matched = make_matched()
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    record = make_record("doctor", "neutral00", "engineer")
    # A neutral pool word missing a covariate key should trip the coverage guard if selected.
    # 11 words: the bridge is filtered out and the fill is 10, so the whole remaining pool is
    # selected whatever the permutation and the broken word is always reached.
    pool = make_neutral_pool(11)
    pool[1] = make_word("neutral01", "neutral", drop_covariate="length")

    with pytest.raises(AssertionError, match="covariate"):
        compose_probe_words(
            record, matched, pool, 0, loaded_index_for(
                matched, target, stereo), rng=Random(0)
        )


def test_probe_neutral_fill_is_not_alphabetical() -> None:
    """The probe neutral fill had the same defect as the control boards, just diluted.

    Only 10 of 25 probe cards are neutral and randomize_positions scatters them, so the contiguous
    alphabetical run was easy to miss on a probe board while being obvious on an all-neutral
    control. Same window, same pool, same fix - asserted separately so neither path can regress
    alone.
    """
    matched = make_matched()
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    record = make_record("doctor", "neutral00", "engineer")
    pool = make_neutral_pool(200)

    words = compose_probe_words(
        record, matched, pool, 0, loaded_index_for(
            matched, target, stereo), rng=Random(7)
    )
    fill = [w.text for w in words[15:]]
    assert len(fill) == 10
    assert fill != sorted(fill)

    candidates = sorted(w.text for w in pool if w.text != "neutral00")
    doubled = candidates + candidates
    windows = [doubled[i:i + len(fill)] for i in range(len(candidates))]
    assert sorted(fill) not in windows


def test_probe_neutral_fill_depends_on_the_rng() -> None:
    matched = make_matched()
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    record = make_record("doctor", "neutral00", "engineer")
    pool = make_neutral_pool(200)

    def fill(seed: int) -> list[str]:
        words = compose_probe_words(
            record, matched, pool, 0, loaded_index_for(
                matched, target, stereo), rng=Random(seed)
        )
        return [w.text for w in words[15:]]

    assert fill(1) == fill(1)
    assert fill(1) != fill(999)


# compose_control_words


def test_control_returns_25_distinct_neutrals() -> None:
    words = compose_control_words(make_neutral_pool(40), 0, rng=Random(1))
    assert len(words) == 25
    assert len({w.text for w in words}) == 25
    assert all(w.gender_category == "neutral" for w in words)


def test_control_same_index_and_seed_is_identical() -> None:
    pool = make_neutral_pool(40)
    first = compose_control_words(pool, 3, rng=Random(1))
    second = compose_control_words(pool, 3, rng=Random(1))
    assert [w.text for w in first] == [w.text for w in second]


def test_control_selection_depends_on_the_rng() -> None:
    """The discriminating test: the seed must actually reach the neutral pick.

    This inverts the previous contract. The window used to be taken over the text-sorted pool, so
    the rng was accepted and immediately discarded (`del rng`) and any two seeds gave the same 25
    words. A seed that cannot change the selection cannot decorrelate it from alphabetical order.
    """
    pool = make_neutral_pool(40)
    first = compose_control_words(pool, 3, rng=Random(1))
    second = compose_control_words(pool, 3, rng=Random(999))
    assert [w.text for w in first] != [w.text for w in second]


def test_control_selection_is_not_alphabetical() -> None:
    """The 25 words must not be a contiguous alphabetical run of the pool."""
    pool = make_neutral_pool(200)
    texts = [w.text for w in compose_control_words(pool, 0, rng=Random(7))]
    assert texts != sorted(texts)

    all_sorted = sorted(w.text for w in pool)
    # ...nor any contiguous (wrapping) slice of the sorted pool, whatever the offset.
    doubled = all_sorted + all_sorted
    windows = [doubled[i:i + len(texts)] for i in range(len(all_sorted))]
    assert sorted(texts) not in windows


def test_control_pool_coverage_is_preserved() -> None:
    """Boards sharing one permutation still consume the pool near-exhaustively.

    This is the property that forces ONE bank-level permutation rather than a per-board seed. With
    a per-board seed the window offset becomes meaningless (each board indexes into its own
    permutation) and selection degenerates into independent sampling, stranding a large fraction of
    the pool. Asserted explicitly so that regression is loud.
    """
    pool = make_neutral_pool(332)
    seed = 4242
    boards = [
        [w.text for w in compose_control_words(pool, i, rng=Random(seed))] for i in range(14)
    ]

    used = Counter(text for board in boards for text in board)
    assert len(used) == len(pool), "every pool word must still be used at least once"
    # 14 * 25 = 350 slots over 332 words -> exactly 18 words wrap into a second board.
    assert sum(1 for count in used.values() if count > 1) == 18
    assert max(used.values()) == 2


def test_control_includes_oov_neutral() -> None:
    # Exactly BOARD_WORD_COUNT words once the OOV is appended, so the window covers the whole pool
    # whatever the permutation. (It used to rely on "aaa_tipi" sorting first, which only held while
    # selection ran over the text-sorted pool.)
    pool = make_neutral_pool(24)
    oov = Word(
        text="aaa_tipi",
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


# pair_selection_overlap


def test_overlap_dispersion_beats_consecutive_window() -> None:
    # Realistic case: 15 eligible pairs, 8 chosen per board, over the 8 career indices.
    # Old consecutive window: stride = ceil(15/8) = 2 -> worst-case intersection n_pairs-stride = 6.
    report = pair_selection_overlap(range(15), range(8), n_pairs=8)
    assert report["n_board_pairs"] == 28  # C(8, 2)
    assert report["mean_intersection"] < 6  # demonstrably better dispersion
    assert 0.0 <= report["mean_jaccard"] <= 1.0
    assert 0.0 <= report["max_jaccard"] <= 1.0


def test_overlap_identical_board_indices_is_full() -> None:
    report = pair_selection_overlap(range(15), [3, 3], n_pairs=8)
    assert report["n_board_pairs"] == 1
    assert report["max_jaccard"] == 1.0
    assert report["mean_jaccard"] == 1.0
    assert report["max_intersection"] == 8


def test_overlap_single_board_is_well_defined_empty() -> None:
    report = pair_selection_overlap(range(15), [0], n_pairs=8)
    assert report["n_board_pairs"] == 0
    assert report["mean_jaccard"] == 0.0
    assert report["max_jaccard"] == 0.0
    assert report["mean_intersection"] == 0.0
    assert report["max_intersection"] == 0
    assert report["selections"] == {0: sorted(report["selections"][0])}


def test_overlap_mirrors_compose_selection() -> None:
    # The diagnostic must report the same indices compose_probe_words actually places.
    matched = make_matched(15)
    target, stereo = make_word("doctor", "male"), make_word("engineer", "male")
    index = loaded_index_for(matched, target, stereo)
    pool = make_neutral_pool(20)
    record = make_record("doctor", "neutral00", "engineer")

    eligible = list(range(len(matched.treatment)))
    # mirror the compose default (n_pairs=6) so the diagnostic reports the placed pairs.
    report = pair_selection_overlap(eligible, [2], n_pairs=6)

    words = compose_probe_words(record, matched, pool, 2, index, rng=Random(0))
    placed_treatment = {
        w.text for w in words[3:15] if w.gender_category == "male"}
    expected = {matched.treatment[i].text for i in report["selections"][2]}
    assert placed_treatment == expected
