"""Interactive semi-automatic CLI. Exposes main() and run_dilemma_flow().

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

from pathlib import Path

from board_generator import dilemma_flow
from board_generator.arbiter import DEFAULT_CONSENSUS, Arbiter, load_consensus
from board_generator.lexicon import Specification, Word, load_words

# Tool input data defaults (under board_generator/resources/, configurable).
DEFAULT_WORDS_DIR = Path(__file__).resolve(
).parent.parent / "resources" / "words"
DEFAULT_SUBTLEX_PATH = (
    Path(__file__).resolve().parent.parent /
    "resources" / "frequencies" / "subtlex_us.csv"
)


def main() -> None:
    """Run the interactive board-bank generation flow (manual steps stay manual)."""
    raise NotImplementedError


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
