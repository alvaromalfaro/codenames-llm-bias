"""Offline tests for the semi-automatic dilemma flow core (board_generator.dilemma_flow).

Every test runs on hand-engineered ScriptedEncoder geometries wrapped in plain Arbiters; the real
backend, φ*, HF revisions and the network are NEVER touched. Cosines are read off 2D unit vectors so
cos(a, b) is the dot product and can be checked by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from board_generator import dilemma_flow
from board_generator.arbiter import Arbiter, ArbiterRef
from board_generator.dilemma_flow import DilemmaSession
from board_generator.lexicon import GenderCategory, Specification, Word

from ._stub_encoders import ScriptedEncoder


def make_word(text: str, gender: GenderCategory = "neutral") -> Word:
    """Build a board-eligible Word directly, bypassing CSV/WordNet loading."""
    return Word(
        text=text,
        gender_category=gender,
        word_kind="common",
        source="test",
        weat_set=("weat-6",),
        dom_pos=None,
        ambiguous_pos=False,
        covariates={"subtlex_freq": 1.0, "length": float(
            len(text)), "wordnet_polysemy": 1.0},
        specification="gender-career",
    )


def arbiter(model_id: str, vectors: dict[str, NDArray[np.float64]]) -> Arbiter:
    """Wrap a ScriptedEncoder over the given geometry in a pinned Arbiter (no real HF rev)."""
    return Arbiter(ref=ArbiterRef(model_id, "rev-test"), encoder=ScriptedEncoder(vectors))


# target on the x-axis; each candidate's cos to it is just its x component (unit vectors).
_RANK_GEOMETRY = {
    "nurse": np.array([1.0, 0.0]),
    "hospital": np.array([1.0, 0.0]),  # cos 1.0
    "ward": np.array([0.8, 0.6]),  # cos 0.8
    "clinic": np.array([0.6, 0.8]),  # cos 0.6
    "dress": np.array([0.0, 1.0]),  # cos 0.0
}


def _ranking_session() -> DilemmaSession:
    phi = arbiter("stub/rank", _RANK_GEOMETRY)
    return DilemmaSession(phi_star=phi, consensus=[phi], specification="gender-career")


# ranking


def test_rank_neutral_respects_k_sort_and_excludes_target() -> None:
    session = _ranking_session()
    target = make_word("nurse")
    neutral = [make_word(t) for t in ("nurse", "clinic", "hospital", "ward")]

    ranked = session.rank_neutral(target, neutral, k=2)

    assert [w.text for w, _ in ranked] == [
        "hospital", "ward"]  # desc by cos, target dropped
    assert [cos for _, cos in ranked] == pytest.approx([1.0, 0.8])
    assert len(ranked) <= 2


def test_rank_tie_breaks_on_text_ascending() -> None:
    phi = arbiter(
        "stub/tie",
        {
            "nurse": np.array([1.0, 0.0]),
            "beta": np.array([0.6, 0.8]),
            "alpha": np.array([0.6, -0.8]),
        },
    )
    session = DilemmaSession(phi_star=phi, consensus=[
                             phi], specification="gender-career")
    target = make_word("nurse")

    ranked = session.rank_neutral(
        target, [make_word("beta"), make_word("alpha")], k=2)

    assert [w.text for w, _ in ranked] == ["alpha", "beta"]
    assert ranked[0][1] == pytest.approx(ranked[1][1])


def test_stereo_ranking_never_includes_gender_incongruent_words() -> None:
    phi = arbiter(
        "stub/cong",
        {
            "manager": np.array([1.0, 0.0]),
            "boss": np.array([0.9, np.sqrt(1.0 - 0.81)]),  # male, congruent
            # female, incongruent
            "nurse": np.array([0.95, np.sqrt(1.0 - 0.95**2)]),
            # neutral, incongruent
            "table": np.array([0.99, np.sqrt(1.0 - 0.99**2)]),
        },
    )
    session = DilemmaSession(phi_star=phi, consensus=[
                             phi], specification="gender-career")
    target = make_word("manager", gender="male")
    loaded = [
        target,
        make_word("boss", gender="male"),
        make_word("nurse", gender="female"),
        make_word("table", gender="neutral"),
    ]

    ranked = session.rank_stereo(target, loaded, k=1)

    # only the male word, despite higher cosines
    assert [w.text for w, _ in ranked] == ["boss"]
    assert all(w.gender_category == "male" for w, _ in ranked)


def test_thin_pool_warns_when_available_below_k() -> None:
    session = _ranking_session()
    target = make_word("nurse")

    with pytest.warns(UserWarning, match="< k=8"):
        session.rank_neutral(
            target, [make_word("ward"), make_word("clinic")], k=8)


# verify / accept / reject accounting


def _agreeing(model_id: str) -> Arbiter:
    # c_n (0.8) > c_s (0.6) => satisfies Eq. 4.1.
    return arbiter(
        model_id,
        {
            "nurse": np.array([1.0, 0.0]),
            "hospital": np.array([0.8, 0.6]),
            "dress": np.array([0.6, 0.8]),
        },
    )


def _accept_geometry(model_id: str) -> Arbiter:
    # nurse-hospital (neutral) closer than nurse-gown (stereo): passes. nurse-dress: fails.
    return arbiter(
        model_id,
        {
            "nurse": np.array([1.0, 0.0]),
            "hospital": np.array([0.8, 0.6]),  # cos 0.8 (neutral)
            # cos 0.7 (stereo, passes)
            "gown": np.array([0.7, np.sqrt(1.0 - 0.49)]),
            # cos 0.9 (stereo, fails)
            "dress": np.array([0.9, np.sqrt(1.0 - 0.81)]),
        },
    )


def test_accept_path_sets_consensus_ok_and_serializes(tmp_path: Path) -> None:
    consensus = [_agreeing("stub/a"), _agreeing("stub/b")]
    session = DilemmaSession(
        phi_star=consensus[0], consensus=consensus, specification="gender-career"
    )

    dilemma = session.attempt(
        make_word("nurse"), make_word("hospital"), make_word("dress"))
    assert dilemma.consensus_ok is True

    record = session.build_record(dilemma)
    assert record.attempts_count == 1
    assert record.rejected_attempts == []

    path = dilemma_flow.write_record(record, tmp_path)
    assert path.exists()
    assert path.name == "dilemma_gender-career_nurse.json"


def test_t9_rejected_attempt_is_recorded(tmp_path: Path) -> None:
    # First stereo (dress) FAILS, second (gown) PASSES under all arbiters.
    consensus = [_accept_geometry("stub/a"), _accept_geometry("stub/b")]
    session = DilemmaSession(
        phi_star=consensus[0], consensus=consensus, specification="gender-science"
    )

    first = session.attempt(make_word("nurse"), make_word(
        "hospital"), make_word("dress"))
    assert first.consensus_ok is False

    second = session.attempt(
        make_word("nurse"), make_word("hospital"), make_word("gown"))
    assert second.consensus_ok is True

    record = session.build_record(second)

    assert record.attempts_count == 2
    assert len(record.rejected_attempts) == 1
    rejected = record.rejected_attempts[0]
    assert rejected.consensus_ok is False
    assert rejected.stereotypical_bridge == "dress"
    # The rejected attempt keeps its per-arbiter cosines (search pressure stays auditable).
    assert all(s.cos_target_stereo >
               s.cos_target_neutral for s in rejected.arbiter_scores)
    assert record.accepted is second
    assert record.accepted.stereotypical_bridge == "gown"


def test_distinctness_collision_raises() -> None:
    session = _ranking_session()
    with pytest.raises(AssertionError):
        session.attempt(make_word("nurse"), make_word(
            "hospital"), make_word("hospital"))


def test_attempt_cap_raises_after_too_many_rejections() -> None:
    consensus = [_accept_geometry("stub/a")]
    session = DilemmaSession(
        phi_star=consensus[0], consensus=consensus, specification="gender-career", attempt_cap=1
    )
    with pytest.raises(RuntimeError, match="attempt cap"):
        session.attempt(make_word("nurse"), make_word(
            "hospital"), make_word("dress"))


# artifact: id, overwrite guard, round-trip re-verify


def _accepted_record(spec: Specification = "gender-career") -> dilemma_flow.DilemmaRecord:
    consensus = [_agreeing("stub/a"), _agreeing("stub/b")]
    session = DilemmaSession(
        phi_star=consensus[0], consensus=consensus, specification=spec)
    dilemma = session.attempt(
        make_word("nurse"), make_word("hospital"), make_word("dress"))
    return session.build_record(dilemma)


def test_record_id_is_spec_underscore_target() -> None:
    assert dilemma_flow.record_id(
        "gender-career", "nurse") == "gender-career_nurse"


def test_existing_artifact_is_not_silently_overwritten(tmp_path: Path) -> None:
    record = _accepted_record()
    dilemma_flow.write_record(record, tmp_path)

    with pytest.raises(FileExistsError):
        dilemma_flow.write_record(record, tmp_path)

    # Explicit overwrite is allowed.
    dilemma_flow.write_record(record, tmp_path, overwrite=True)


def test_round_trip_and_reverify_reproduces_consensus_ok(tmp_path: Path) -> None:
    consensus = [_agreeing("stub/a"), _agreeing("stub/b")]
    record = _accepted_record()

    path = dilemma_flow.write_record(record, tmp_path)
    loaded = dilemma_flow.read_record(path)

    assert loaded.target == record.target
    assert loaded.accepted.consensus_ok == record.accepted.consensus_ok
    assert loaded.attempts_count == record.attempts_count

    reverified = dilemma_flow.reverify(loaded, consensus)
    assert reverified.consensus_ok == record.accepted.consensus_ok is True


def test_pinned_arbiters_match_injected_refs() -> None:
    phi = _agreeing("stub/primary")
    other = _agreeing("stub/other")
    session = DilemmaSession(
        phi_star=phi, consensus=[phi, other], specification="gender-career"
    )
    dilemma = session.attempt(
        make_word("nurse"), make_word("hospital"), make_word("dress"))
    record = session.build_record(dilemma)

    assert record.arbiters_consensus == [str(phi.ref), str(other.ref)]
    assert record.arbiters_primary == str(phi.ref)
    assert record.arbiters_consensus == [
        "stub/primary@rev-test", "stub/other@rev-test"]
