"""Lexical composition: select the 25 words of a board.

This module picks which words appear on a board; it assigns no key-card roles or grid positions
and performs no serialization. It is deterministic and fully offline: no embeddings, no primary
arbiter φ*, no Hugging Face, no network.

Two board types are composed:

- Probe (:func:compose_probe_words): the 3-word dilemma block (target / stereotypical bridge /
  neutral bridge), plus a balanced loaded fill of whole PSM pairs (6 male + 6 female), plus a
  neutral fill - 3 + 12 + 10 = 25 words.
- Control (:func:compose_control_words): 25 distinct neutral words, no dilemma.

Every candidate pool is sorted by text before selection, so the output never depends on set/dict
iteration order. Sorting is the canonical starting order, never the selection order: the loaded PSM
pairs are chosen by a per-board deterministic permutation (see :func:_permutation_pick /
:func:pair_selection_overlap), and the neutral pool is permuted by :func:shuffled_neutral_pool
before the cyclic window is taken. Same inputs + same board index + same rng seed -> identical
25-word list, in identical order.

Why the neutral pool is shuffled: the window is contiguous, so windowing a TEXT-SORTED pool made a
board's neutral words a contiguous alphabetical run (control-000 got the A words, control-001 the B
words, ...), correlating board composition with alphabetical position. Shuffling the pool once
breaks that correlation while leaving the window - and therefore the near-exhaustive coverage the
window guarantees - intact.
"""

from __future__ import annotations

import itertools
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
    rng: Random,
    n_pairs: int = 6,
) -> list[Word]:
    """Compose the 25 words of a probe board.

    Layout (deterministic order): the 3-word dilemma block, then 2 * n_pairs loaded fill words
    (whole PSM pairs -> 50/50 male/female with inherited covariate balance), then the remaining
    neutral fill. With n_pairs=6 that is 3 + 12 + 10 = 25.

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
        rng: seeded RNG fixing the neutral-fill pool permutation, exactly as for control boards
            (:func:compose_control_words). The neutral fill is only 10 of 25 cards here, so the
            alphabetical run it used to produce was diluted by the 15 loaded/dilemma words rather
            than absent.
        n_pairs: number of whole PSM pairs to place (default 6 -> 12 loaded words).

    Pair selection: let eligible be the PSM pair indices whose treatment and control words are both
    distinct from the target and stereo. Board j takes the first n_pairs of a per-board
    deterministic permutation of eligible, seeded by j alone (see :func:_permutation_pick). This
    disperses pair sets across boards so probe boards are not near-clones, while staying
    byte-reproducible. The old consecutive-window scheme overlapped heavily; dispersion is no longer
    guaranteed analytically but is instead MEASURED via :func:pair_selection_overlap.

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

    selected = _permutation_pick(eligible, board_index_in_spec, n_pairs)
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
    # Permute the whole pool, then drop the bridge: filtering first would make the permutation
    # depend on which word the bridge is, so probe boards would no longer share one ordering.
    neutral_shuffled = [
        w for w in shuffled_neutral_pool(neutral_pool, rng) if w.text != neutral_bridge.text
    ]
    neutral_fill = _cyclic_window(
        neutral_shuffled, board_index_in_spec, neutral_count)

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

    Selection is a cyclic window of 25 words over the shuffled neutral pool
    (:func:shuffled_neutral_pool), keyed by board_index. Deterministic and independent of dict/set
    iteration order, but no longer correlated with alphabetical position: windowing the text-sorted
    pool directly gave each board a contiguous alphabetical run. Reuse of neutral words across the
    control boards is expected and allowed (≈332 neutrals vs 14x25 = 350 slots); only each single
    board's 25 words must be distinct.

    Args:
        neutral_pool: the gender-neutral words. OOV neutrals (subtlex_freq=None) are kept.
        board_index: 0-based index of this control board; keys the cyclic window.
        rng: seeded RNG that fixes the pool permutation. Load-bearing: the same rng state yields the
            same permutation and hence the same board. Every board in a bank must be handed an rng
            at the same state (a fresh Random(pool_seed)) so they share one permutation and the
            window keeps consuming the pool near-exhaustively.

    Returns:
        Exactly 25 :class:Word objects, all with distinct text.
    """
    pool_shuffled = shuffled_neutral_pool(neutral_pool, rng)
    words = _cyclic_window(pool_shuffled, board_index, BOARD_WORD_COUNT)
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


def _permutation_pick(eligible: Sequence[int], board_index: int, n_pairs: int) -> list[int]:
    """Pick up to n_pairs eligible pair indices via a per-board deterministic permutation.

    Returns the eligible values (not positions). When the pool is smaller than or equal to n_pairs
    the whole eligible list is returned (the caller warns).

    Selection draws n_pairs distinct indices from a permutation of eligible seeded by board_index
    alone (a fresh Random(board_index) per call). Seeding on the board index and not on the board
    seed keeps the dispersion pattern stable and independent of the board's role/position seed, so
    it is reproducible byte-for-byte (same board_index -> same selection).
    """
    length = len(eligible)
    if length == 0:
        return []
    if length <= n_pairs:
        return list(eligible)
    k = min(n_pairs, length)
    return Random(board_index).sample(list(eligible), k)


def pair_selection_overlap(
    eligible: Sequence[int], board_indices: Sequence[int], n_pairs: int = 8
) -> dict:
    """Report inter-board overlap of selected PSM pair sets for a range of boards.

    Mirrors the selection used by :func:compose_probe_words exactly (same :func:_permutation_pick),
    so the operator can measure dispersion before committing a bank (e.g. over the 8 career indices)
    and turn "do the probe boards look too similar?" into an objective number. Pure and
    deterministic: no I/O, no embeddings, no primary arbiter.

    For every unordered pair of boards the Jaccard overlap is |A∩B| / |A∪B|; an empty union (both
    selections empty, i.e. no eligible pairs) is reported as 0.0 by convention. With fewer than two
    boards there are no board pairs, so the aggregates are a well-defined empty result (0.0 / 0 with
    n_board_pairs == 0).

    Returns a dict with:
        - selections: {board_index: sorted selected pair indices}. Keyed by int; callers serializing
          to JSON should stringify the keys.
        - mean_jaccard / max_jaccard: pairwise Jaccard overlap across all board pairs.
        - mean_intersection / max_intersection: raw |A∩B| across all board pairs.
        - n_board_pairs: number of unordered board pairs compared.
    """
    selections = {
        bi: frozenset(_permutation_pick(eligible, bi, n_pairs)) for bi in board_indices
    }

    jaccards: list[float] = []
    intersections: list[int] = []
    for a, b in itertools.combinations(board_indices, 2):
        set_a, set_b = selections[a], selections[b]
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccards.append(inter / union if union else 0.0)
        intersections.append(inter)

    n_board_pairs = len(jaccards)
    return {
        "selections": {bi: sorted(s) for bi, s in selections.items()},
        "mean_jaccard": sum(jaccards) / n_board_pairs if n_board_pairs else 0.0,
        "max_jaccard": max(jaccards) if jaccards else 0.0,
        "mean_intersection": sum(intersections) / n_board_pairs if n_board_pairs else 0.0,
        "max_intersection": max(intersections) if intersections else 0,
        "n_board_pairs": n_board_pairs,
    }


def shuffled_neutral_pool(neutral_pool: Sequence[Word], rng: Random) -> list[Word]:
    """The neutral pool in a reproducible pseudo-random order: sort by text, then permute.

    Sorting first is load-bearing, not redundant with the shuffle. random.Random.shuffle permutes
    whatever order it is handed, so an input whose order varied with dict/set iteration would yield
    a different permutation from the same seed. Sorting pins the canonical starting order, which is
    what makes the permutation reproducible.

    The caller owns the rng and must hand every board an rng at the SAME state (a fresh
    Random(pool_seed) per call), so all boards share one permutation of the pool. That shared
    permutation is what preserves the cyclic window's coverage guarantee: consecutive boards take
    disjoint windows of one ordering, so the pool is consumed near-exhaustively. Seeding this
    per-board instead would give each board its own permutation, making the window offset
    meaningless and degenerating selection into independent per-board sampling - which strands a
    large fraction of the pool and is exactly what this scheme avoids.

    The rng must NOT be the board's role/position stream (board.assemble_board's Random(seed)); see
    bank.build_bank for the derivation that keeps the two disjoint.
    """
    pool_sorted = sorted(neutral_pool, key=lambda w: w.text)
    rng.shuffle(pool_sorted)
    return pool_sorted


def _cyclic_window(pool: Sequence[Word], board_index: int, count: int) -> list[Word]:
    """Take count consecutive words from pool (wrapping), offset by board_index.

    pool must already be in its canonical selection order (see :func:shuffled_neutral_pool for the
    neutral pools). The words are distinct as long as count <= len(pool).
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
