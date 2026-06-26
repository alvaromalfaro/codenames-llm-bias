"""Board assembly and serialization.

Defines the JSON schema dataclasses and writes each board file plus the bank-level
balance_report.json to the configurable output directory (default ../data/boards/, the single
coupling point with the platform).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from board_generator.arbiter import ConsensusSpec
from board_generator.balancing import BalanceReport
from board_generator.lexicon import GenderCategory, Specification, Word
from board_generator.roles import (
    KeyCard,
    KeycardAudit,
    Role,
    assign_roles,
    build_keycard_audit,
    validate_keycard,
)

# Default output directory relative to board_generator/ (configurable).
DEFAULT_OUTPUT_DIR = Path("../data/boards")

# Board lexical type.
BoardType = Literal["control", "probe"]


@dataclass(frozen=True, slots=True)
class Grid:
    """The 5x5 grid dimensions."""

    rows: int = 5
    cols: int = 5


@dataclass(frozen=True, slots=True)
class WordEntry:
    """One serialized grid cell: word, both perspective roles, and covariates."""

    text: str
    index: int
    role_a: Role
    role_b: Role
    gender_category: GenderCategory
    source: str
    covariates: dict[str, float]


@dataclass(frozen=True, slots=True)
class ArbiterScore:
    """Per-arbiter evaluation for a dilemma."""

    arbiter: str  # "model@rev"
    cos_target_neutral: float
    cos_target_stereo: float
    satisfies_eq_4_1: bool


@dataclass(frozen=True, slots=True)
class Dilemma:
    """Probe-board dilemma block; None for control boards."""

    target: str
    neutral_bridge: str
    stereotypical_bridge: str
    # AND of the per-arbiter satisfies_eq_4_1 flags
    consensus_ok: bool
    arbiter_scores: list[ArbiterScore]


@dataclass(frozen=True, slots=True)
class ArbiterSet:
    """The arbiters block: consensus list + primary φ* (both as "model@rev")."""

    consensus: list[str]
    primary: str


@dataclass(frozen=True, slots=True)
class Board:
    """A full board record matching."""

    board_id: str
    type: BoardType
    specification: Specification
    seed: int
    arbiters: ArbiterSet
    grid: Grid
    words: list[WordEntry]
    dilemma: Dilemma | None
    keycard_audit: KeycardAudit


def arbiter_set_from_spec(spec: ConsensusSpec) -> ArbiterSet:
    """Build the arbiters block from a validated ConsensusSpec (φ* ∈ consensus)."""
    return ArbiterSet(
        consensus=[str(ref) for ref in spec.encoders],
        primary=str(spec.primary),
    )


def validate_board_grid(board: Board) -> bool:
    """Verify 25 unique words on a 5x5 grid with indices 0..24."""
    if board.grid.rows != 5 or board.grid.cols != 5:
        return False
    expected = board.grid.rows * board.grid.cols
    if len(board.words) != expected:
        return False
    texts = [w.text for w in board.words]
    if len(set(texts)) != expected:
        return False
    indices = sorted(w.index for w in board.words)
    return indices == list(range(expected))


def randomize_positions(
    words: Sequence[Word],
    keycard: KeyCard,
    dilemma_words: Sequence[Word] | None,
    rng: random.Random,
) -> list[Word]:
    """Place the 25 words onto the 25 grid positions, deriving the order from the board seed.

    Returns the words in position order: result[i] is the word at grid index i, so the caller pairs
    result[i] with keycard.role_a[i] / keycard.role_b[i].

    For a probe board the 3 dilemma_words must land on positions whose LLM-perspective role is
    "agent" (keycard.role_b[i] == "agent"), per the IAE (the model is clue-giver and the 3 dilemma
    words are its unrevealed agents). The other 22 words are placed by a GENDER-BLIND uniform
    permutation: gender_category is never consulted, so position/role does not correlate with
    gender. For a control board (dilemma_words empty/None) all 25 words are placed by the
    gender-blind permutation with no reserved positions.

    All randomness derives from rng in a fixed consumption order - (1) rng.sample picks the
    dilemma's agent positions (probe only), then (2) rng.shuffle permutes the remaining words. Same
    rng state + same inputs -> identical placement. Reordering these two draws would change the
    output, so the order is part of the contract.

    The rng handed in by assemble_board has ALREADY been advanced by assign_roles, so positions are
    drawn from the post-roles state of the single shared stream (see assemble_board for the full
    consumption order).
    """
    dilemma_words = list(dilemma_words or [])
    dilemma_texts = {w.text for w in dilemma_words}
    remaining = [w for w in words if w.text not in dilemma_texts]

    result: list[Word | None] = [None] * len(words)
    chosen: list[int] = []
    if dilemma_words:
        # The 9 LLM-perspective agent positions, in grid-index order.
        agent_b = [i for i, role in enumerate(
            keycard.role_b) if role == "agent"]
        # (1) deterministic pick of len(dilemma_words) of the 9 agent positions.
        chosen = rng.sample(agent_b, len(dilemma_words))
        for word, pos in zip(dilemma_words, chosen, strict=True):
            result[pos] = word

    open_positions = [p for p in range(len(words)) if result[p] is None]
    # (2) single gender-blind permutation of the remaining words; gender is never consulted.
    rng.shuffle(remaining)
    for pos, word in zip(open_positions, remaining, strict=True):
        result[pos] = word

    placed = [w for w in result if w is not None]
    assert len(placed) == len(
        words), "every position must be filled exactly once"
    return placed


def assemble_board(
    board_id: str,
    board_type: BoardType,
    specification: Specification,
    seed: int,
    words: list[Word],
    consensus: ConsensusSpec,
    dilemma: Dilemma | None,
) -> Board:
    """Assemble a fully-validated Board from its parts.

    This function OWNS a single rng (random.Random(seed)) consumed in a load-bearing order: it first
    builds the key card, then places the words, so roles and positions derive from DISJOINT segments
    of one stream rather than from the shared prefix of two independent streams.

        rng = random.Random(seed)
        keycard = assign_roles(rng)                                  # consumes the first segment
        placed  = randomize_positions(words, keycard, dilemma_words, rng)  # consumes the NEXT one

    assign_roles must consume before randomize_positions, so reordering the two draws would change
    the board. Building both off the same seed makes the bank reproducible byte-for-byte. Callers
    that need the key card can read board.keycard_audit; it is deliberately not a parameter.

    Covariates pass through verbatim, including None for OOV neutrals; imputation is balancing's
    job. Validates the assembled board (grid, key card, and - for probe boards - that the dilemma
    words sit at LLM-agent positions) before returning.

    Raises:
        ValueError: a dilemma word is absent from words, or a post-assembly invariant fails.
    """
    rng = random.Random(seed)
    keycard = assign_roles(rng)

    if dilemma is not None:
        index = {w.text: w for w in words}
        # Fixed order, mirroring composition's dilemma_block (target, stereo, neutral_bridge).
        dilemma_texts = (
            dilemma.target, dilemma.stereotypical_bridge, dilemma.neutral_bridge)
        try:
            dilemma_words = [index[text] for text in dilemma_texts]
        except KeyError as exc:
            raise ValueError(
                f"dilemma word {exc.args[0]!r} not among the board words") from exc
    else:
        dilemma_words = []

    placed = randomize_positions(words, keycard, dilemma_words, rng)

    entries = [
        WordEntry(
            text=word.text,
            index=i,
            role_a=keycard.role_a[i],
            role_b=keycard.role_b[i],
            gender_category=word.gender_category,
            source=word.source,
            # Pass covariates through verbatim, including None for OOV neutrals.
            # WordEntry.covariates is typed dict[str, float], so the None is a deliberate,
            # documented gap rather than a mismatch to impute away here.
            covariates=dict(word.covariates),  # type: ignore[arg-type]
        )
        for i, word in enumerate(placed)
    ]

    board = Board(
        board_id=board_id,
        type=board_type,
        specification=specification,
        seed=seed,
        arbiters=arbiter_set_from_spec(consensus),
        grid=Grid(),
        words=entries,
        dilemma=dilemma,
        keycard_audit=build_keycard_audit(keycard, placed, dilemma_words),
    )

    if not validate_board_grid(board):
        raise ValueError(
            f"assembled board {board_id!r} failed grid validation")
    if not validate_keycard(keycard):
        raise ValueError(
            f"assembled board {board_id!r} has an illegal key card")
    if dilemma is not None:
        dilemma_texts_set = {w.text for w in dilemma_words}
        for entry in entries:
            if entry.text in dilemma_texts_set and entry.role_b != "agent":
                raise ValueError(
                    f"dilemma word {entry.text!r} is not at an LLM-agent position "
                    f"(role_b={entry.role_b!r})"
                )

    return board


def to_json_dict(board: Board) -> dict[str, Any]:
    """Serialize a Board to the JSON structure (contract with the platform reader)."""
    raise NotImplementedError


def write_board(board: Board, out_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write one board file under out_dir (default ../data/boards/)."""
    raise NotImplementedError


def write_balance_report(report: BalanceReport, out_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write the bank-level balance_report.json under out_dir."""
    raise NotImplementedError
