"""Dilemma construction.

Semi-automatic. This module ONLY ranks candidates and verifies the dilemma condition. The target / 
neutral-bridge / stereotypical-bridge selections are MANUAL expert choices and are deliberately NOT
automated here. The tool assists; the human decides.

Ranking uses the primary arbiter φ* alone (manual choice follows); the Eq. 4.1 gate uses
the full consensus set and accepts a triple only if the inequality holds for all arbiters (consensus_ok).
"""

from __future__ import annotations

from collections.abc import Sequence

from board_generator.arbiter import Arbiter
from board_generator.board import Dilemma
from board_generator.lexicon import Word


def rank_neutral_bridges(
    target: Word, neutral_words: Sequence[Word], phi_star: Arbiter, k: int
) -> list[tuple[Word, float]]:
    """Rank k neutral words by descending cos(φ*) to the target (AUTO)."""
    raise NotImplementedError


def rank_stereotypical_bridges(
    target: Word, congruent_words: Sequence[Word], phi_star: Arbiter, k: int
) -> list[tuple[Word, float]]:
    """Rank k gender-congruent words by descending cos(φ*) to the target (AUTO)."""
    raise NotImplementedError


def verify_eq_4_1(
    target: Word,
    neutral_bridge: Word,
    stereo_bridge: Word,
    consensus: Sequence[Arbiter],
) -> Dilemma:
    """Verify Eq. 4.1 under every consensus arbiter (AUTO).

    Accept iff cos(target, neutral) >= cos(target, stereo) for all arbiters (consensus_ok). Records 
    the per-arbiter cosines and flags.
    """
    raise NotImplementedError
