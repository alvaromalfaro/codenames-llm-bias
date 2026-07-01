"""CLI with two subcommands: bank and dilemma. Exposes main() and run_dilemma_flow().

The tool has two operations of opposite natures, one per subcommand:
- bank    - batch, offline build of the board bank from a manifest (board_generator.bank).
- dilemma - interactive build of one dilemma (needs φ*/HF; run_dilemma_flow).

Orchestrates pool loading, covariate balancing, lexical composition, role assignment, the
semi-automatic dilemma loop (the manual target/bridge selections stay manual), position
randomization and serialization. Determinism: every random draw derives from the recorded board
seed.

This is the I/O layer: it prompts for the three manual selections (target / neutral bridge /
stereotypical bridge) and presents the ranked candidates. All ranking, verification, reject/retry
accounting and serialization live in the φ*-agnostic core (board_generator.dilemma_flow), which is
where the offline tests exercise the logic. The flow here is deliberately not unit-tested; it only
wires prompts to the core.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from board_generator import bank, board, dilemma_flow
from board_generator.arbiter import DEFAULT_CONSENSUS, Arbiter, load_consensus
from board_generator.lexicon import Specification, Word, load_words

# Tool input data defaults (under board_generator/resources/, configurable).
DEFAULT_WORDS_DIR = Path(__file__).resolve(
).parent.parent / "resources" / "words"
DEFAULT_SUBTLEX_PATH = (
    Path(__file__).resolve().parent.parent /
    "resources" / "frequencies" / "subtlex_us.csv"
)


def main(argv: list[str] | None = None) -> None:
    """Dispatch to a subcommand (bank or dilemma); pure argparse wiring, no logic.

    The two flows have opposite natures: bank is a batch, offline build from a manifest, and dilemma
    is the interactive build of one dilemma. A missing subcommand exits non-zero.
    """
    parser = argparse.ArgumentParser(
        prog="board-generator",
        description="Generate Codenames Duet boards instrumented for gender-bias measurement.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_bank_subparser(subparsers)
    _add_dilemma_subparser(subparsers)

    args = parser.parse_args(argv)
    if args.command == "bank":
        _run_bank(args)
    elif args.command == "dilemma":
        _run_dilemma(args)


def _add_bank_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the bank subcommand with the bank-build arguments (no logic beyond argparse)."""
    parser = subparsers.add_parser(
        "bank",
        help="Build the fixed Codenames Duet board bank from a manifest.",
        description="Build the fixed Codenames Duet board bank from a manifest.",
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="path to the bank manifest JSON"
    )
    parser.add_argument("--words-dir", type=Path, default=DEFAULT_WORDS_DIR)
    parser.add_argument("--subtlex-path", type=Path,
                        default=DEFAULT_SUBTLEX_PATH)
    parser.add_argument(
        "--dilemmas-dir", type=Path, default=dilemma_flow.DEFAULT_DILEMMA_DIR
    )
    parser.add_argument("--out-dir", type=Path,
                        default=board.DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and validate the bank but write nothing",
    )


def _add_dilemma_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the dilemma subcommand with the interactive arguments."""
    parser = subparsers.add_parser(
        "dilemma",
        help="Build one dilemma interactively; needs the primary arbiter φ*.",
        description="Build one dilemma interactively; needs the primary arbiter φ*.",
    )
    parser.add_argument(
        "--spec",
        required=True,
        choices=["gender-career", "gender-science"],
        help="the dilemma specification",
    )
    parser.add_argument("--words-dir", type=Path, default=DEFAULT_WORDS_DIR)
    parser.add_argument("--subtlex-path", type=Path,
                        default=DEFAULT_SUBTLEX_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=dilemma_flow.DEFAULT_DILEMMA_DIR,
        help="directory for the dilemma_<id>.json artifact",
    )
    parser.add_argument("--k", type=int, default=dilemma_flow.DEFAULT_K)
    parser.add_argument("--attempt-cap", type=int, default=None)


def _run_bank(args: argparse.Namespace) -> None:
    """Build the board bank from a manifest (thin I/O around bank.build_bank).

    Wiring only: read the manifest and the dilemma artifacts it lists, call the pure core, then
    write the boards and the bank-level balance report (skipped under --dry-run). All composition,
    seeding and validation live in board_generator.bank.
    """
    manifest = bank.load_manifest(args.manifest)
    records = [
        dilemma_flow.read_record(args.dilemmas_dir / filename)
        for filename in manifest.dilemmas
    ]
    boards, report, warnings = bank.build_bank(
        manifest,
        records,
        words_dir=args.words_dir,
        subtlex_path=args.subtlex_path,
    )

    if not args.dry_run:
        for board_record in boards:
            board.write_board(board_record, args.out_dir)
        board.write_balance_report(report, args.out_dir)

    _print_bank_summary(
        boards, warnings, dry_run=args.dry_run, out_dir=args.out_dir)


def _run_dilemma(args: argparse.Namespace) -> None:
    """Drive the interactive dilemma flow and print the written artifact path (wiring only)."""
    path = run_dilemma_flow(
        args.spec,
        words_dir=args.words_dir,
        subtlex_path=args.subtlex_path,
        out_dir=args.out_dir,
        k=args.k,
        attempt_cap=args.attempt_cap,
    )
    print(f"\nWrote dilemma artifact to {path}")


def _print_bank_summary(
    boards: list[board.Board],
    warnings: list[str],
    *,
    dry_run: bool,
    out_dir: Path,
) -> None:
    """Print a one-screen bank summary: counts, per-spec probe breakdown and any warnings."""
    probes = [b for b in boards if b.type == "probe"]
    controls = [b for b in boards if b.type == "control"]
    per_spec = Counter(b.specification for b in probes)

    print(
        f"\nBuilt {len(boards)} board(s): {len(probes)} probe + {len(controls)} control (50/50).")
    for spec in sorted(per_spec):
        print(f"  probe {spec}: {per_spec[spec]}")

    if warnings:
        print(
            f"\nWarnings ({len(warnings)} board(s), descriptive - not a gate):")
        for line in warnings:
            print(f"  {line}")

    if dry_run:
        print("\n--dry-run: validated, wrote nothing.")
    else:
        print(f"\nWrote boards + balance_report.json to {out_dir}")


# Presentation / prompt helpers (no logic)


def _covariate_str(word: Word) -> str:
    """Compact one-line covariate summary for a candidate row."""
    keys = ("subtlex_freq", "length", "wordnet_polysemy")
    return ", ".join(f"{key}={word.covariates.get(key)}" for key in keys)


def _present_candidates(title: str, ranked: list[tuple[Word, float]]) -> None:
    """Print a ranked candidate table: cos(φ*), gender_category and covariates."""
    print(f"\n{title}")
    for i, (word, cos) in enumerate(ranked):
        print(
            f"  [{i:2d}] {word.text:<18} cos(φ*)={cos:.4f}  "
            f"gender={word.gender_category:<7}  {_covariate_str(word)}"
        )


def _prompt_word(prompt: str, pool: list[Word]) -> Word:
    """Prompt the author to type a word from pool; reprompts until a known text is entered."""
    by_text = {w.text: w for w in pool}
    while True:
        choice = input(f"{prompt}: ").strip().lower()
        if choice in by_text:
            return by_text[choice]
        print(f"  '{choice}' is not in the pool; choose one of the listed words.")


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}


def run_dilemma_flow(
    specification: Specification,
    *,
    words_dir: Path = DEFAULT_WORDS_DIR,
    subtlex_path: Path = DEFAULT_SUBTLEX_PATH,
    out_dir: Path = dilemma_flow.DEFAULT_DILEMMA_DIR,
    k: int = dilemma_flow.DEFAULT_K,
    attempt_cap: int | None = None,
) -> Path:
    """Drive Stage D interactively for one dilemma and write the verified DilemmaRecord.

    Loads the word pools and the consensus arbiters (network: pinned HF revisions), then walks the
    six steps. The three selections are manual; the ranking and the Eq. 4.1 gate are auto. On a
    failing triple the author re-picks any of target/neutral/stereo (not just the stereo).
    """
    load_result = load_words(words_dir, subtlex_path)
    loaded = load_result.words

    consensus = load_consensus(DEFAULT_CONSENSUS)
    phi_star = _primary_arbiter(consensus)
    session = dilemma_flow.DilemmaSession(
        phi_star=phi_star,
        consensus=consensus,
        specification=specification,
        attempt_cap=attempt_cap,
    )

    # Step 1 [Manual]: pick the target from the loaded male/female words of this specification.
    target_pool = [
        w
        for w in loaded
        if w.specification == specification and w.gender_category in {"male", "female"}
    ]
    target_title = f"Target pool ({specification}, male/female):"
    _present_candidates(target_title, [(w, 0.0) for w in target_pool])
    target = _prompt_word("Pick the target word", target_pool)

    while True:
        # Step 2 [Auto] -> 3 [Manual]: neutral bridge.
        neutral_ranked = session.rank_neutral(target, loaded, k=k)
        _present_candidates(
            "Neutral-bridge candidates (ranked by cos(φ*)):", neutral_ranked)
        neutral = _prompt_word("Pick the neutral bridge", [
                               w for w, _ in neutral_ranked])

        # Step 4 [Auto] -> 5 [Manual]: stereotypical bridge (same metadata incl. absolute cosine).
        stereo_ranked = session.rank_stereo(target, loaded, k=k)
        _present_candidates(
            "Stereotypical-bridge candidates (ranked by cos(φ*)):", stereo_ranked)
        stereo = _prompt_word("Pick the stereotypical bridge", [
                              w for w, _ in stereo_ranked])

        # Step 6 [Auto]: verify Eq. 4.1 under the consensus.
        dilemma = session.attempt(target, neutral, stereo)
        if dilemma.consensus_ok:
            break

        print("\n  REJECTED - Eq. 4.1 fails under the consensus:")
        for score in dilemma.arbiter_scores:
            mark = "ok" if score.satisfies_eq_4_1 else "VIOLATED"
            print(
                f"    {score.arbiter}: cos(neutral)={score.cos_target_neutral:.4f} "
                f">= cos(stereo)={score.cos_target_stereo:.4f}? {mark}"
            )
        if _confirm("Re-pick the target as well?"):
            _present_candidates(target_title, [(w, 0.0) for w in target_pool])
            target = _prompt_word("Pick the target word", target_pool)
        print("Re-selecting bridges...")

    record = session.build_record(dilemma)
    try:
        return dilemma_flow.write_record(record, out_dir)
    except FileExistsError as exc:
        print(f"\n  {exc}")
        if not _confirm("Overwrite the existing artifact?"):
            raise
        return dilemma_flow.write_record(record, out_dir, overwrite=True)


def _primary_arbiter(consensus: list[Arbiter]) -> Arbiter:
    """Return the loaded arbiter designated as the primary φ* (DEFAULT_CONSENSUS.primary)."""
    for arbiter in consensus:
        if arbiter.ref == DEFAULT_CONSENSUS.primary:
            return arbiter
    raise ValueError(
        "primary φ* not found among the loaded consensus arbiters")


if __name__ == "__main__":
    main()
