"""Board assembly and serialization.

Defines the JSON schema dataclasses and writes each board file plus the bank-level balance_report.json
to the configurable output directory (default ../data/boards/, the single coupling point with the
platform).
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
from board_generator.roles import KeyCard, KeycardAudit, Role

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


def randomize_positions(words: Sequence[Word], rng: random.Random) -> list[Word]:
    """Randomize the 25 grid positions, deriving the order from the board seed."""
    raise NotImplementedError


def assemble_board(
    board_id: str,
    board_type: BoardType,
    specification: Specification,
    seed: int,
    words: list[Word],
    keycard: KeyCard,
    consensus: ConsensusSpec,
    dilemma: Dilemma | None,
) -> Board:
    """Assemble a fully-validated Board from its parts."""
    raise NotImplementedError


def to_json_dict(board: Board) -> dict[str, Any]:
    """Serialize a Board to the JSON structure (contract with the platform reader)."""
    raise NotImplementedError


def write_board(board: Board, out_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write one board file under out_dir (default ../data/boards/)."""
    raise NotImplementedError


def write_balance_report(report: BalanceReport, out_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write the bank-level balance_report.json under out_dir."""
    raise NotImplementedError
