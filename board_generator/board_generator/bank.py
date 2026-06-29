"""Master bank loop: the φ*-agnostic core that assembles the full board bank.

This is the top of the funnel. Given a versionable manifest (a master seed plus an ordered list of
pre-built dilemma artifacts) and the parsed :class:DilemmaRecord's, :func:build_bank emits the
entire bank of boards (one probe per dilemma + an equal number of all-neutral controls) plus the
bank-level balance report. It is deterministic, fully OFFLINE and never touches the primary arbiter
φ*: dilemmas are consumed verbatim from disk and their consensus_ok is trusted from the artifact,
never recomputed.

The single source of truth for the bank is the manifest: the same manifest over the same word pools
yields the same bank, byte for byte. Every per-board random draw derives from a per-board seed that
is itself derived from (master_seed, board_id) via :func:derive_board_seed - distinct per board and
stable under manifest reordering (a board's seed depends only on its id, not its position in the
list). The bank-level balance runs once, seeded by the master seed.

This module reads the tool's own word pools and writes nothing; serialization lives in the I/O layer
(cli.main) which calls :func:build_bank and then board.write_board / board.write_balance_report. The
only platform coupling is those output files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

from board_generator.arbiter import DEFAULT_CONSENSUS
from board_generator.balancing import BalanceReport, MatchedSubset, run_balancing
from board_generator.board import Board, assemble_board
from board_generator.composition import compose_control_words, compose_probe_words
from board_generator.dilemma_flow import DilemmaRecord
from board_generator.lexicon import Specification, load_words

# Whole PSM pairs placed on each probe board: 3 dilemma + 2*6 loaded + 10 neutral = 25 cards. Fixed
# (small PSM pool decision); exposed as a parameter only to keep build_bank testable.
DEFAULT_N_PAIRS = 6


@dataclass(frozen=True, slots=True)
class Manifest:
    """The bank's complete, versionable definition.

    master_seed is the single source of truth for the bank's determinism. dilemmas is an ordered
    list of artifact FILENAMES (resolved relative to a configurable dilemmas dir by the I/O layer);
    the order assigns each probe board its index within its specification.
    """

    master_seed: int
    dilemmas: list[str]


def load_manifest(path: Path) -> Manifest:
    """Read and validate a manifest JSON file: {"master_seed": int, "dilemmas": [filename, ...]}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} must be a JSON object")
    seed = data.get("master_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"manifest {path} 'master_seed' must be an integer")
    dilemmas = data.get("dilemmas")
    if not isinstance(dilemmas, list) or not all(isinstance(d, str) for d in dilemmas):
        raise ValueError(
            f"manifest {path} 'dilemmas' must be a list of filenames")
    return Manifest(master_seed=seed, dilemmas=list(dilemmas))


def derive_board_seed(master_seed: int, board_id: str) -> int:
    """Per-board seed derived from (master_seed, board_id): deterministic, distinct, order-stable.

    A board's seed depends only on its id, so reordering the manifest never changes a given board's
    seed. The 8-byte SHA-256 prefix gives a stable 64-bit integer across runs and platforms, unlike
    Python's salted hash.
    """
    digest = hashlib.sha256(f"{master_seed}:{board_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_bank(
    manifest: Manifest,
    records: list[DilemmaRecord],
    *,
    words_dir: Path,
    subtlex_path: Path,
    n_pairs: int = DEFAULT_N_PAIRS,
) -> tuple[list[Board], BalanceReport, list[str]]:
    """Assemble the full board bank (probe + control) from a manifest and its dilemma records.

    The bank is 2 * P boards for P = len(records): one probe per dilemma (in manifest order, indexed
    within its specification) and P all-neutral controls. 50/50 holds by construction. records are
    passed already parsed (the I/O layer resolves and reads the artifact filenames) so this core is
    unit-testable with fixture records; manifest.dilemmas is consumed only to name artifacts in
    error messages and is checked to be parallel to records.

    Returns the boards, the bank-level balance report (run once, seeded by master_seed), and a list
    of human-readable warnings (role/gender dependence is descriptive, not a gate). Raises on any
    structural violation or a stale-artifact word reference.
    """
    if len(manifest.dilemmas) != len(records):
        raise ValueError(
            f"manifest lists {len(manifest.dilemmas)} dilemma(s) but {len(records)} record(s) "
            "were provided"
        )

    words = load_words(words_dir, subtlex_path).words
    balance = run_balancing(words, seed=manifest.master_seed)
    matched: dict[Specification, MatchedSubset] = {
        m.specification: m for m in balance.matched
    }
    neutral_pool = [w for w in words if w.gender_category == "neutral"]
    loaded_index = {w.text: w for w in words}

    boards: list[Board] = []

    # Probe boards: one per record, in manifest order, indexed within their specification.
    index_in_spec: dict[Specification, int] = {}
    for filename, record in zip(manifest.dilemmas, records, strict=True):
        if not record.accepted.consensus_ok:
            raise ValueError(
                f"dilemma artifact {filename!r} has consensus_ok=False; a rejected dilemma must "
                "not appear in the manifest"
            )
        spec = record.specification
        idx = index_in_spec.get(spec, 0)
        index_in_spec[spec] = idx + 1
        board_id = f"probe-{spec}-{idx:03d}"
        if spec not in matched:
            raise ValueError(
                f"no balanced subset for specification {spec!r} (board {board_id!r})"
            )
        seed = derive_board_seed(manifest.master_seed, board_id)
        try:
            words_25 = compose_probe_words(
                record, matched[spec], neutral_pool, idx, loaded_index, n_pairs=n_pairs
            )
            board = assemble_board(
                board_id, "probe", spec, seed, words_25, DEFAULT_CONSENSUS, record.accepted
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"failed to build probe board {board_id!r} from artifact {filename!r}: {exc}"
            ) from exc
        boards.append(board)

    # Control boards: P all-neutral boards, no dilemma.
    pairs = len(records)
    for i in range(pairs):
        board_id = f"control-{i:03d}"
        seed = derive_board_seed(manifest.master_seed, board_id)
        words_25 = compose_control_words(
            neutral_pool, board_index=i, rng=Random(seed))
        boards.append(
            assemble_board(
                board_id,
                "control",
                # Controls carry no specification; it is serialized as JSON null. The frozen
                # assemble_board signature types this non-optional, so the None is deliberate.
                None,  # type: ignore[arg-type]
                seed,
                words_25,
                DEFAULT_CONSENSUS,
                None,
            )
        )

    warnings = validate_bank_invariants(boards, pairs)
    return boards, balance.report, warnings


def validate_bank_invariants(boards: list[Board], expected_pairs: int) -> list[str]:
    """Hard-check structural invariants (FAIL) and collect descriptives (REPORT).

    Raises ValueError count / 50-50, board_id uniqueness, or control purity / probe consensus
    violations. Returns a list of human-readable warnings for any board whose key card audit flags
    role/gender dependence.
    """
    probes = [b for b in boards if b.type == "probe"]
    controls = [b for b in boards if b.type == "control"]

    # 2*P boards, 50/50 by construction.
    if len(boards) != 2 * expected_pairs:
        raise ValueError(
            f"bank has {len(boards)} boards, expected {2 * expected_pairs} (2 * {expected_pairs})"
        )
    if len(probes) != len(controls):
        raise ValueError(
            f"bank is not 50/50: {len(probes)} probe board(s) vs {len(controls)} control board(s)"
        )
    if len(probes) != expected_pairs:
        raise ValueError(
            f"bank has {len(probes)} probe board(s), expected {expected_pairs}"
        )

    # board_id uniqueness.
    ids = [b.board_id for b in boards]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate board_id(s): {duplicates}")

    # control purity and probe dilemma/consensus presence.
    for board in boards:
        if board.type == "control":
            if board.dilemma is not None:
                raise ValueError(
                    f"control board {board.board_id!r} must not carry a dilemma"
                )
            if not all(w.gender_category == "neutral" for w in board.words):
                raise ValueError(
                    f"control board {board.board_id!r} has non-neutral card(s)"
                )
        else:  # probe
            if board.dilemma is None:
                raise ValueError(
                    f"probe board {board.board_id!r} is missing its dilemma"
                )
            if not board.dilemma.consensus_ok:
                raise ValueError(
                    f"probe board {board.board_id!r} carries a dilemma with consensus_ok=False"
                )

    # report (do not fail) any board whose roles depend on gender.
    return [
        f"Board {b.board_id!r} role assignment is NOT independent of gender_category"
        for b in boards
        if not b.keycard_audit.role_gender_independent
    ]
