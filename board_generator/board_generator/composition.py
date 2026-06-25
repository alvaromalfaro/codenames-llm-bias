"""Lexical composition: select the 25 words of a board.

This module picks which words appear on a board; it assigns no key-card roles or grid positions
and performs no serialization. It is deterministic and fully offline: no embeddings, no primary
arbiter ``φ*``, no Hugging Face, no network.

Two board types are composed:

- Probe (:func:compose_probe_words): the 3-word dilemma block (target / stereotypical bridge /
  neutral bridge), plus a balanced loaded fill of whole PSM pairs (8 male + 8 female), plus a
  neutral fill — 3 + 16 + 6 = 25 words.
- Control (:func:compose_control_words): 25 distinct neutral words, no dilemma.

Selection is index-derived (cyclic windows keyed by the board index), and every candidate pool is
sorted by text before windowing, so the output never depends on set/dict iteration order. Same
inputs + same board index -> identical 25-word list, in identical order.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from random import Random

from board_generator.balancing import MatchedSubset
from board_generator.dilemma_flow import DilemmaRecord
from board_generator.lexicon import COVARIATE_KEYS, Word

BOARD_WORD_COUNT = 25
DILEMMA_WORD_COUNT = 3


def compose_probe_words(
    record: DilemmaRecord,
    matched_subset: MatchedSubset,
    neutral_pool: Sequence[Word],
    board_index_in_spec: int,
    loaded_index: Mapping[str, Word],
    *,
    n_pairs: int = 8,
) -> list[Word]:
    """Compose the 25 words of a probe board.

    Layout (deterministic order): the 3-word dilemma block, then 2 * n_pairs loaded fill words
    (whole PSM pairs -> 50/50 male/female with inherited covariate balance), then the remaining
    neutral fill. With n_pairs=8 that is 3 + 16 + 6 = 25.

    Args:
        record: the accepted dilemma; target / stereotypical_bridge / neutral_bridge are stored as
            text and resolved to :class:Word here.
        matched_subset: the pair-aligned PSM output for this specification; treatment[i] and
            control[i] form one covariate-balanced male/female pair.
        neutral_pool: the gender-neutral words available for the neutral fill (and to resolve the
            neutral bridge). OOV neutrals (subtlex_freq=None) are valid and kept.
        board_index_in_spec: this board's 0-based index within its specification; keys the cyclic
            windows for pair and neutral selection.
        loaded_index: {w.text: w} over the full loaded-word pool. The dilemma's loaded words
            (target / stereo) may be absent from matched_subset under the lax inclusion policy,
            so they are resolved through this index rather than the matched subset.
        n_pairs: number of whole PSM pairs to place (default 8 -> 16 loaded words).

    Pair rotation: let eligible be the PSM pair indices whose treatment and control words are both
    distinct from the target and stereo. With L = len(eligible) and
    stride = max(1, ceil(L / n_pairs)), board j takes the n_pairs eligible positions
    [(j*stride + m) mod L for m in range(n_pairs)] (de-duplicated). The stride spreads consecutive
    boards across the pool so their pair sets overlap by at most n_pairs - stride pairs while
    staying byte-reproducible.

    Returns:
        Exactly 25 :class:Word objects, all with distinct text.

    Raises:
        ValueError: a dilemma word cannot be resolved.
        AssertionError: a post-condition (count, uniqueness, covariate coverage, dilemma presence)
            is violated.
    """
    treatment = matched_subset.treatment
    control = matched_subset.control
    assert len(treatment) == len(
        control), "PSM pairs must be aligned (equal treatment/control len)"

    target = _resolve(record.target, loaded_index, "target")
    stereo = _resolve(record.stereotypical_bridge,
                      loaded_index, "stereotypical_bridge")
    neutral_index = {w.text: w for w in neutral_pool}
    neutral_bridge = _resolve(record.neutral_bridge,
                              neutral_index, "neutral_bridge")
    dilemma_block = [target, stereo, neutral_bridge]

    # Exclude any pair containing a loaded dilemma word, defensively regardless of policy.
    excluded_texts = {target.text, stereo.text}
    eligible = [
        i
        for i in range(len(treatment))
        if treatment[i].text not in excluded_texts and control[i].text not in excluded_texts
    ]

    selected = _cyclic_pick(eligible, board_index_in_spec, n_pairs)
    if len(eligible) < n_pairs:
        warnings.warn(
            f"thin balanced pool: {len(eligible)} eligible pair(s) < n_pairs={n_pairs}; "
            f"placing all eligible pairs",
            stacklevel=2,
        )

    loaded_fill: list[Word] = []
    for i in selected:
        loaded_fill.append(treatment[i])
        loaded_fill.append(control[i])

    neutral_count = BOARD_WORD_COUNT - DILEMMA_WORD_COUNT - 2 * len(selected)
    neutral_sorted = sorted(
        (w for w in neutral_pool if w.text != neutral_bridge.text), key=lambda w: w.text
    )
    neutral_fill = _cyclic_window(
        neutral_sorted, board_index_in_spec, neutral_count)

    words = dilemma_block + loaded_fill + neutral_fill
    _assert_board_words(words)
    for dilemma_word in dilemma_block:
        assert (
            sum(1 for w in words if w.text == dilemma_word.text) == 1
        ), f"dilemma word {dilemma_word.text!r} must appear exactly once"
    return words


def compose_control_words(
    neutral_pool: Sequence[Word],
    board_index: int,
    *,
    rng: Random,
) -> list[Word]:
    """Compose the 25 words of a control board: 25 distinct neutral words, no dilemma.

    Selection is a cyclic window of 25 words over the text-sorted neutral pool, keyed by
    board_index - deterministic and independent of dict/set iteration order. Reuse of neutral words
    across the 15 control boards is expected and allowed (≈332 neutrals vs 15x25 = 375 slots); only
    each single board's 25 words must be distinct.

    Args:
        neutral_pool: the gender-neutral words. OOV neutrals (subtlex_freq=None) are kept.
        board_index: 0-based index of this control board; keys the cyclic window.
        rng: seeded RNG, accepted for interface symmetry with the future caller and reserved for any
            residual tie-break. The window itself is index-derived, so the output does not depend on
            rng today; it is kept to match the planned call site, not dead by oversight.

    Returns:
        Exactly 25 :class:Word objects, all with distinct text.
    """
    del rng  # reserved; window is index-derived for byte-reproducibility
    pool_sorted = sorted(neutral_pool, key=lambda w: w.text)
    words = _cyclic_window(pool_sorted, board_index, BOARD_WORD_COUNT)
    assert len(
        words) == BOARD_WORD_COUNT, f"expected {BOARD_WORD_COUNT} words, got {len(words)}"
    assert len({w.text for w in words}) == len(
        words), "control board words must be distinct"
    return words


def _resolve(text: str, index: Mapping[str, Word], role: str) -> Word:
    """Resolve a dilemma selection (stored as text) to its :class:Word."""
    word = index.get(text)
    if word is None:
        raise ValueError(
            f"dilemma {role} {text!r} not found in its candidate pool")
    return word


def _cyclic_pick(eligible: Sequence[int], board_index: int, n_pairs: int) -> list[int]:
    """Pick up to n_pairs eligible pair indices via the strided cyclic window.

    Returns the eligible values (not positions), de-duplicated in selection order. When the pool is
    smaller than n_pairs the whole eligible list is returned (the caller warns).
    """
    length = len(eligible)
    if length == 0:
        return []
    if length <= n_pairs:
        return list(eligible)
    stride = max(1, math.ceil(length / n_pairs))
    picked: list[int] = []
    seen: set[int] = set()
    for m in range(n_pairs):
        value = eligible[(board_index * stride + m) % length]
        if value not in seen:
            seen.add(value)
            picked.append(value)
    return picked


def _cyclic_window(pool: Sequence[Word], board_index: int, count: int) -> list[Word]:
    """Take count consecutive words from pool (wrapping), offset by board_index.

    pool must already be text-sorted. The words are distinct as long as count <= len(pool).
    """
    length = len(pool)
    assert count <= length, f"cannot take {count} distinct words from a pool of {length}"
    offset = (board_index * count) % length
    return [pool[(offset + m) % length] for m in range(count)]


def _assert_board_words(words: Sequence[Word]) -> None:
    """Guard the shared board post-conditions: count, uniqueness, covariate coverage."""
    assert len(
        words) == BOARD_WORD_COUNT, f"expected {BOARD_WORD_COUNT} words, got {len(words)}"
    assert len({w.text for w in words}) == len(
        words), "board words must be distinct"
    for word in words:
        missing = [key for key in COVARIATE_KEYS if key not in word.covariates]
        assert not missing, f"word {word.text!r} missing covariate(s): {missing}"
