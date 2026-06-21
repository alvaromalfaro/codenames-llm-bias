"""Dilemma construction - candidate ranking and Eq. 4.1 verification.

Semi-automatic. This module ONLY ranks candidates and verifies the dilemma condition. The target /
neutral-bridge / stereotypical-bridge selections are MANUAL expert choices and are deliberately NOT
automated here. The tool assists; the human decides.

Ranking uses the primary arbiter φ* alone (manual choice follows); the Eq. 4.1 gate uses the full
consensus set and accepts a triple only if the inequality holds for all arbiters (consensus_ok).
"""

from __future__ import annotations

from collections.abc import Sequence

from board_generator.arbiter import Arbiter
from board_generator.board import ArbiterScore, Dilemma
from board_generator.lexicon import Word


def _rank(
    target: Word, candidates: Sequence[Word], phi_star: Arbiter, k: int
) -> list[tuple[Word, float]]:
    """Rank candidates by descending cos(φ*) to the target; return the top k (AUTO).

    Embeds the target once, excludes the target itself if present, and breaks cosine ties on
    Word.text ascending so equal cosines yield a stable, reproducible order.
    """
    target_vec = phi_star.embed(target.text)
    scored = [
        (word, phi_star.cos(target_vec, phi_star.embed(word.text)))
        for word in candidates
        if word.text != target.text
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0].text))
    return scored[:k]


def rank_neutral_bridges(
    target: Word, neutral_words: Sequence[Word], phi_star: Arbiter, k: int
) -> list[tuple[Word, float]]:
    """Rank k neutral words by descending cos(φ*) to the target (AUTO)."""
    return _rank(target, neutral_words, phi_star, k)


def rank_stereotypical_bridges(
    target: Word, congruent_words: Sequence[Word], phi_star: Arbiter, k: int
) -> list[tuple[Word, float]]:
    """Rank k gender-congruent words by descending cos(φ*) to the target (AUTO)."""
    return _rank(target, congruent_words, phi_star, k)


def verify_eq_4_1(
    target: Word,
    neutral_bridge: Word,
    stereo_bridge: Word,
    consensus: Sequence[Arbiter],
) -> Dilemma:
    """Verify Eq. 4.1 under every consensus arbiter (AUTO).

    Accept iff cos(target, neutral) >= cos(target, stereo) for all arbiters (consensus_ok). The
    inequality is non-strict, with no margin, so an exact tie passes. Records the per-arbiter
    cosines (full float64) and flags; rounding/casing is board.to_json_dict's job, not here.
    """
    scores: list[ArbiterScore] = []
    consensus_ok = True
    for arbiter in consensus:
        target_vec = arbiter.embed(target.text)
        cos_target_neutral = arbiter.cos(
            target_vec, arbiter.embed(neutral_bridge.text))
        cos_target_stereo = arbiter.cos(
            target_vec, arbiter.embed(stereo_bridge.text))
        satisfies = cos_target_neutral >= cos_target_stereo
        consensus_ok = consensus_ok and satisfies
        scores.append(
            ArbiterScore(
                arbiter=str(arbiter.ref),
                cos_target_neutral=cos_target_neutral,
                cos_target_stereo=cos_target_stereo,
                satisfies_eq_4_1=satisfies,
            )
        )
    return Dilemma(
        target=target.text,
        neutral_bridge=neutral_bridge.text,
        stereotypical_bridge=stereo_bridge.text,
        consensus_ok=consensus_ok,
        arbiter_scores=scores,
    )
