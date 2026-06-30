"""Semi-automatic dilemma flow (φ*-agnostic core).

This module wires the frozen auto building blocks (board_generator.dilemma.rank_neutral_bridges /
rank_stereotypical_bridges / verify_eq_4_1) around the manual selection stops, accumulates rejected
attempts for auditability (search pressure against the consensus gate must stay visible), and
serializes a verified dilemma as an intermediate artifact.

The core is φ*-agnostic and offline-testable. It receives an INJECTED primary arbiter and a
consensus sequence of arbiters (not a ConsensusSpec, whose construction would reject stub arbiters
that carry no real HF revision). There is no input()/print() here - the interactive prompts and the
candidate presentation live in the thin cli.py I/O layer.

The accepted dilemma is a board.Dilemma, embedded verbatim inside the new DilemmaRecord wrapper.
Stage B later extracts record.accepted unchanged - no reconversion. The record carries NO seed: the
dilemma is seed-independent.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from board_generator import board
from board_generator.arbiter import Arbiter
from board_generator.dilemma import (
    rank_neutral_bridges,
    rank_stereotypical_bridges,
    verify_eq_4_1,
)
from board_generator.lexicon import Specification, Word

# Intermediate artifacts live under board_generator/ (isolation: never data/boards/). resources/ is
# the tool's own data folder, a sibling of the import package. Configurable everywhere; this is only
# the default.
DEFAULT_DILEMMA_DIR = Path(__file__).resolve(
).parent.parent / "resources" / "dilemmas"

# Default candidate count for the ranked presentations; configurable.
DEFAULT_K = 8


@dataclass(frozen=True, slots=True)
class DilemmaRecord:
    """Construction wrapper around an accepted board.Dilemma.

    board.Dilemma is frozen and has no slot for rejected attempts, so the audit trail lives here.
    The accepted dilemma is stored verbatim; rejected_attempts records every triple that failed the
    consensus gate (each with consensus_ok=False and its per-arbiter cosines), making the search
    pressure against the gate auditable. Carries no seed - the dilemma is seed-independent.
    """

    specification: Specification
    target: str
    neutral_bridge: str
    stereotypical_bridge: str
    accepted: board.Dilemma
    rejected_attempts: list[board.Dilemma]
    attempts_count: int
    # Pinned arbiters in "model@rev" form: the full consensus set and the primary φ*.
    arbiters_consensus: list[str]
    arbiters_primary: str


# Pure pool helpers (the core filters; it never trusts the caller)


def neutral_pool(neutral_words: Sequence[Word], target: Word) -> list[Word]:
    """Neutral-bridge candidate pool: gender-neutral words, target excluded by text."""
    return [
        w for w in neutral_words if w.gender_category == "neutral" and w.text != target.text
    ]


def congruent_pool(loaded_words: Sequence[Word], target: Word) -> list[Word]:
    """Stereotypical-bridge candidate pool: words gender-congruent with the target, target excluded.

    Congruence is filtered HERE rather than trusting the caller (red line): only words whose
    gender_category matches the target's survive.
    """
    return [
        w
        for w in loaded_words
        if w.gender_category == target.gender_category and w.text != target.text
    ]


def _warn_if_thin(pool: Sequence[Word], k: int, kind: str) -> None:
    """Warn when the available pool is thinner than the requested k (varied dilemmas need depth)."""
    if len(pool) < k:
        warnings.warn(
            f"only {len(pool)} {kind} candidate(s) available (< k={k}); presenting all of them - "
            "the loaded subset may be too thin for varied dilemmas",
            stacklevel=3,
        )


# Session: ranks, verifies, accumulates rejected attempts


@dataclass
class DilemmaSession:
    """Stateful orchestrator around the frozen rankers/verifier for one dilemma construction.

    Injected with the primary arbiter φ* (ranking) and the consensus arbiters (the Eq. 4.1 gate);
    both are plain Arbiters so the session is constructible in offline tests with stub encoders.
    Accumulates every rejected attempt across re-selections. The I/O layer drives the manual
    re-selection loop and inspects each attempt's consensus_ok; this core never prompts.
    """

    phi_star: Arbiter
    consensus: Sequence[Arbiter]
    specification: Specification
    attempt_cap: int | None = None
    rejected: list[board.Dilemma] = field(default_factory=list, init=False)

    def rank_neutral(
        self, target: Word, neutral_words: Sequence[Word], k: int = DEFAULT_K
    ) -> list[tuple[Word, float]]:
        """Rank up to k neutral-bridge candidates by cos(φ*) (AUTO)."""
        pool = neutral_pool(neutral_words, target)
        _warn_if_thin(pool, k, "neutral")
        ranked = rank_neutral_bridges(target, pool, self.phi_star, k)
        assert all(w.text != target.text for w,
                   _ in ranked), "target leaked into neutral ranking"
        return ranked

    def rank_stereo(
        self, target: Word, loaded_words: Sequence[Word], k: int = DEFAULT_K
    ) -> list[tuple[Word, float]]:
        """Rank up to k gender-congruent stereo candidates by cos(φ*) (AUTO)."""
        pool = congruent_pool(loaded_words, target)
        _warn_if_thin(pool, k, "stereotypical")
        ranked = rank_stereotypical_bridges(target, pool, self.phi_star, k)
        assert all(
            w.gender_category == target.gender_category for w, _ in ranked
        ), "gender-incongruent word in stereotypical ranking"
        assert all(w.text != target.text for w,
                   _ in ranked), "target leaked into stereo ranking"
        return ranked

    def attempt(self, target: Word, neutral: Word, stereo: Word) -> board.Dilemma:
        """Verify Eq. 4.1 under the consensus for one selected triple (AUTO).

        Asserts the three selections are pairwise distinct, then runs the frozen verifier. A failing
        triple (consensus_ok=False) is appended to the rejected log; the dilemma is returned either
        way so the I/O layer can decide accept vs re-select. Raises if attempt_cap is exceeded.
        """
        texts = {target.text, neutral.text, stereo.text}
        assert len(
            texts) == 3, f"target/neutral/stereo must be distinct, got {sorted(texts)}"

        dilemma = verify_eq_4_1(target, neutral, stereo, self.consensus)
        if not dilemma.consensus_ok:
            self.rejected.append(dilemma)
            if self.attempt_cap is not None and len(self.rejected) >= self.attempt_cap:
                raise RuntimeError(
                    f"attempt cap reached ({self.attempt_cap} rejected attempt(s)); no accepted "
                    "dilemma"
                )
        return dilemma

    def build_record(self, accepted: board.Dilemma) -> DilemmaRecord:
        """Assemble the DilemmaRecord from an accepted dilemma plus the accumulated rejects."""
        assert accepted.consensus_ok, "build_record requires an accepted (consensus_ok) dilemma"
        return DilemmaRecord(
            specification=self.specification,
            target=accepted.target,
            neutral_bridge=accepted.neutral_bridge,
            stereotypical_bridge=accepted.stereotypical_bridge,
            accepted=accepted,
            rejected_attempts=list(self.rejected),
            attempts_count=len(self.rejected) + 1,
            arbiters_consensus=[str(a.ref) for a in self.consensus],
            arbiters_primary=str(self.phi_star.ref),
        )


# Serialization: the intermediate artifact


def record_id(specification: Specification, target: str) -> str:
    """Deterministic record id; target uniqueness per specification becomes a file collision."""
    return f"{specification}_{target}"


def _dilemma_to_dict(dilemma: board.Dilemma) -> dict[str, Any]:
    """Serialize a frozen board.Dilemma at full float64 precision (no rounding here)."""
    return {
        "target": dilemma.target,
        "neutral_bridge": dilemma.neutral_bridge,
        "stereotypical_bridge": dilemma.stereotypical_bridge,
        "consensus_ok": dilemma.consensus_ok,
        "arbiter_scores": [
            {
                "arbiter": s.arbiter,
                "cos_target_neutral": s.cos_target_neutral,
                "cos_target_stereo": s.cos_target_stereo,
                "satisfies_eq_4_1": s.satisfies_eq_4_1,
            }
            for s in dilemma.arbiter_scores
        ],
    }


def _dilemma_from_dict(d: Mapping[str, Any]) -> board.Dilemma:
    """Rebuild a board.Dilemma (and its ArbiterScores) from a serialized dict."""
    return board.Dilemma(
        target=d["target"],
        neutral_bridge=d["neutral_bridge"],
        stereotypical_bridge=d["stereotypical_bridge"],
        consensus_ok=d["consensus_ok"],
        arbiter_scores=[
            board.ArbiterScore(
                arbiter=s["arbiter"],
                cos_target_neutral=s["cos_target_neutral"],
                cos_target_stereo=s["cos_target_stereo"],
                satisfies_eq_4_1=s["satisfies_eq_4_1"],
            )
            for s in d["arbiter_scores"]
        ],
    )


def to_json_dict(record: DilemmaRecord) -> dict[str, Any]:
    """Serialize a DilemmaRecord to a JSON-ready dict (deterministic key order)."""
    return {
        "specification": record.specification,
        "target": record.target,
        "neutral_bridge": record.neutral_bridge,
        "stereotypical_bridge": record.stereotypical_bridge,
        "attempts_count": record.attempts_count,
        "arbiters": {
            "consensus": record.arbiters_consensus,
            "primary": record.arbiters_primary,
        },
        "accepted": _dilemma_to_dict(record.accepted),
        "rejected_attempts": [_dilemma_to_dict(d) for d in record.rejected_attempts],
    }


def from_json_dict(d: Mapping[str, Any]) -> DilemmaRecord:
    """Rebuild a DilemmaRecord from a serialized dict (round-trips with to_json_dict)."""
    return DilemmaRecord(
        specification=d["specification"],
        target=d["target"],
        neutral_bridge=d["neutral_bridge"],
        stereotypical_bridge=d["stereotypical_bridge"],
        accepted=_dilemma_from_dict(d["accepted"]),
        rejected_attempts=[_dilemma_from_dict(
            x) for x in d["rejected_attempts"]],
        attempts_count=d["attempts_count"],
        arbiters_consensus=list(d["arbiters"]["consensus"]),
        arbiters_primary=d["arbiters"]["primary"],
    )


def write_record(
    record: DilemmaRecord,
    out_dir: Path = DEFAULT_DILEMMA_DIR,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the record to out_dir/dilemma_<id>.json. Refuses silent overwrite.

    <id> = f"{specification}_{target}". A pre-existing file raises FileExistsError unless overwrite
    is set explicitly (the I/O layer turns that into a confirm prompt). The JSON is written with a
    deterministic key order so the same record yields the same bytes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / \
        f"dilemma_{record_id(record.specification, record.target)}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"dilemma artifact already exists: {path} (pass overwrite=True to replace)"
        )
    text = json.dumps(to_json_dict(record), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def read_record(path: Path) -> DilemmaRecord:
    """Read a DilemmaRecord back from a dilemma_<id>.json artifact."""
    return from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def _word_shell(text: str) -> Word:
    """A minimal Word carrying only .text, for feeding the frozen verifier on re-verification.

    verify_eq_4_1 reads only Word.text, so the other fields are immaterial to the result; they are
    placeholders. This shell exists solely to re-run the frozen verifier against a record's stored
    selections - it is never placed on a board.
    """
    return Word(
        text=text,
        gender_category="neutral",
        word_kind="common",
        source="reverify",
        weat_set=(),
        dom_pos=None,
        ambiguous_pos=False,
        covariates={},
    )


def reverify(record: DilemmaRecord, consensus: Sequence[Arbiter]) -> board.Dilemma:
    """Re-run the frozen verifier on the record's selections; reproduces consensus_ok.

    Given the same consensus geometry, this reproduces the accepted dilemma's consensus_ok byte for
    byte. Builds text-only Word shells because the record stores selections as text (matching the
    frozen board.Dilemma schema).
    """
    return verify_eq_4_1(
        _word_shell(record.target),
        _word_shell(record.neutral_bridge),
        _word_shell(record.stereotypical_bridge),
        consensus,
    )
