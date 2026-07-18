"""Tests for the frozen ρ reference fixture.

Offline tests use synthetic geometry (no model). Tests that read the committed fixture / re-emitted
sidecar files are offline too (the artifacts are checked in). The one test that recomputes ρ through
the REAL 768-dim φ* is gated on ``integration`` (deselected by default) and skips if the snapshot is
absent. Provenance is asserted (axis/μ̄ /encoder identity), never assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from board_generator.arbiter import DEFAULT_CONSENSUS, Arbiter, ArbiterRef
from board_generator.emit_frame import (
    FrameSnapshotError,
    decode_f64_be_hex,
    frame_content_id,
    read_encoder_recipe,
)
from board_generator.lexicon import load_words
from board_generator.load_filter import (
    AttributeWord,
    build_gender_axis,
    build_mu_bar,
    read_attribute_words,
)
from board_generator.rho_reference import (
    attribute_pole_terms,
    build_rho_reference,
    fixture_path,
    read_board_card_texts,
    weat_association,
)

from ._stub_encoders import HashEncoder, ScriptedEncoder

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOARDS_DIR = _REPO_ROOT / "data" / "boards"
_RESOURCES = Path(__file__).resolve().parents[1] / "resources"
_SIDECAR = _BOARDS_DIR / "measurement_frame.json"
_REFERENCE_BOARD = _BOARDS_DIR / "gender_probe-gender-career-000.json"

_ATTRS = [
    AttributeWord(word="john", gender_pole="male",
                  source="weat", weat_set="weat-6"),
    AttributeWord(word="bill", gender_pole="male",
                  source="weat", weat_set="weat-7"),
    AttributeWord(word="mary", gender_pole="female",
                  source="weat", weat_set="weat-8"),
    AttributeWord(word="anna", gender_pole="female",
                  source="weat", weat_set="weat-6"),
]

_ENCODER = {"name": "stub/phi-star", "revision": "rev-test",
            "pooling": "mean", "normalize": True}


def _phi(dim: int = 768) -> Arbiter:
    return Arbiter(ref=ArbiterRef("stub/phi-star", "rev-test"), encoder=HashEncoder(dim=dim))


# WEAT s(w, A, B), sign + definition (synthetic, exact geometry)
def _u(x: float, y: float) -> np.ndarray:
    v = np.array([x, y], dtype=np.float64)
    return v / np.linalg.norm(v)


def test_weat_association_sign_matches_male_positive() -> None:
    # male terms along +x, female along −x; s>0 means male-associated.
    vectors = {
        "m1": _u(1.0, 0.0), "m2": _u(0.8, 0.6),
        "f1": _u(-1.0, 0.0), "f2": _u(-0.8, 0.6),
        "male_word": _u(0.9, 0.2), "female_word": _u(-0.9, 0.2),
    }
    phi = Arbiter(ref=ArbiterRef("stub", "r"),
                  encoder=ScriptedEncoder(vectors))
    male_vecs = np.vstack([phi.embed("m1"), phi.embed("m2")])
    female_vecs = np.vstack([phi.embed("f1"), phi.embed("f2")])

    s_male = weat_association(phi.embed("male_word"),
                              male_vecs, female_vecs, phi)
    s_female = weat_association(
        phi.embed("female_word"), male_vecs, female_vecs, phi)
    assert s_male > 0.0
    assert s_female < 0.0
    # exact value == mean cos(male) − mean cos(female)
    w = phi.embed("male_word")
    expected = np.mean([phi.cos(w, a) for a in male_vecs]) - np.mean(
        [phi.cos(w, b) for b in female_vecs]
    )
    assert s_male == pytest.approx(expected)


def test_attribute_pole_terms_is_sorted_unique_partition() -> None:
    male, female = attribute_pole_terms(_ATTRS)
    assert male == ["bill", "john"]
    assert female == ["anna", "mary"]


# build_rho_reference, structure + determinism (synthetic 768-dim, no model)
def _build(phi: Arbiter, card_texts: list[str]) -> dict:
    axis = build_gender_axis(_ATTRS, phi)
    mu_bar = build_mu_bar([a.word for a in _ATTRS] + card_texts, phi)
    return build_rho_reference(
        board_id="stub-board",
        card_texts=card_texts,
        attributes=_ATTRS,
        phi_star=phi,
        gender_axis=axis,
        mu_bar=mu_bar,
        encoder=_ENCODER,
        frame_id="deadbeef" * 8,
    )


def test_build_rho_reference_structure() -> None:
    phi = _phi()
    fixture = _build(phi, ["alpha", "beta", "gamma"])
    assert fixture["frame_id"] == "deadbeef" * 8
    assert fixture["board_id"] == "stub-board"
    assert fixture["encoder"] == _ENCODER
    assert "rho_weat_definition" in fixture and "comparison" in fixture
    # rho_weat lexicon is the weat-6 subset of the attributes (john/anna here), not all of _ATTRS.
    assert fixture["rho_weat_lexicon"]["weat_set"] == "weat-6"
    assert fixture["rho_weat_lexicon"]["male_terms"] == ["john"]
    assert fixture["rho_weat_lexicon"]["female_terms"] == ["anna"]
    assert [w["text"] for w in fixture["words"]] == ["alpha", "beta", "gamma"]
    for row in fixture["words"]:
        assert set(row) == {"text", "rho_raw", "rho_cent", "rho_weat"}
        assert all(isinstance(row[k], float)
                   for k in ("rho_raw", "rho_cent", "rho_weat"))


def test_build_rho_reference_is_deterministic() -> None:
    """Guards the freezing code: same encoder + inputs → identical ρ within 1e-12."""
    texts = ["alpha", "beta", "gamma", "delta"]
    first = _build(_phi(), texts)
    second = _build(_phi(), texts)
    for a, b in zip(first["words"], second["words"], strict=True):
        assert a["text"] == b["text"]
        for key in ("rho_raw", "rho_cent", "rho_weat"):
            assert abs(a[key] - b[key]) < 1e-12


# Re-emitted sidecar, content-hash self-consistency + count sanity (offline)
def test_reemitted_sidecar_content_hash_is_self_consistent() -> None:
    """frame_id == hash of its content."""
    sidecar = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    assert frame_content_id(sidecar) == sidecar["frame_id"]
    # created_at + diagnostics are OUTSIDE the hash: tampering them does not change the id.
    tampered = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    tampered["created_at"] = "tampered"
    tampered["mu_bar"]["norm"] = -1.0
    tampered["mu_bar"]["cos_with_axis"] = 9.0
    assert frame_content_id(tampered) == sidecar["frame_id"]


def test_centering_reference_counts_are_arithmetically_consistent() -> None:
    """attr_unique + loaded_unique - collisions == n_reference_items; report the exact numbers."""
    words = load_words(_RESOURCES / "words", _RESOURCES /
                       "frequencies" / "subtlex_us.csv").words
    attrs = read_attribute_words(
        _RESOURCES / "attribute_words" / "gender_attributes.csv")
    loaded = [w for w in words if w.specification is not None]

    attr_lower = {a.word.lower() for a in attrs}
    loaded_lower = {w.text.lower() for w in loaded}
    collisions = attr_lower & loaded_lower
    n_reference = len(attr_lower | loaded_lower)

    sidecar = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    ref = sidecar["mu_bar"]["centering_reference"]
    assert ref["n_reference_items"] == n_reference
    # identity holds without a guessed constant:
    assert len(attr_lower) + len(loaded_lower) - len(collisions) == n_reference

    # loaded pole counts are exhaustive (no neutral-labelled loaded words hidden by the m/f schema).
    neutral_loaded = sum(1 for w in loaded if w.gender_category == "neutral")
    assert neutral_loaded == 0
    loaded_mf = ref["loaded_counts_by_pole"]
    assert loaded_mf["male"] + loaded_mf["female"] == len(loaded)


# Frozen fixture on disk — keyed to the sidecar frame_id, sign agreement (offline: reads artifacts)
def _load_committed_fixture() -> dict:
    sidecar = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    path = fixture_path(sidecar["frame_id"], out_dir=Path(
        __file__).resolve().parent / "fixtures")
    if not path.is_file():
        pytest.skip(
            f"ρ reference fixture not emitted for frame {sidecar['frame_id'][:8]}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_fixture_frame_id_matches_reemitted_sidecar() -> None:
    """The fixture and sidecar frame_id must agree."""
    sidecar = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    fixture = _load_committed_fixture()
    assert fixture["frame_id"] == sidecar["frame_id"]
    assert fixture["encoder"] == sidecar["encoder"]


def test_fixture_rho_weat_is_decoupled_but_male_positive() -> None:
    """The disjoint WEAT-6 lexicon must decouple rho_weat from rho_raw (corr < 1, not const·rho_raw)
    while keeping the male-positive direction (corr > 0). corr ≈ 1 would mean the split failed."""
    fixture = _load_committed_fixture()
    # rho_weat is computed over the disjoint weat-6 names lexicon, 8/8 by pole.
    lexicon = fixture["rho_weat_lexicon"]
    assert lexicon["weat_set"] == "weat-6"
    assert lexicon["counts_by_pole"] == {"male": 8, "female": 8}

    raw = np.array([w["rho_raw"] for w in fixture["words"]])
    weat = np.array([w["rho_weat"] for w in fixture["words"]])
    corr = float(np.corrcoef(raw, weat)[0, 1])
    assert fixture["rho_raw_weat_correlation"] == pytest.approx(corr)
    # Male-positive direction preserved, but NOT collinear (the collinear version was exactly 1.0).
    assert corr > 0.0, "rho_weat lost the male-positive direction"
    assert corr < 0.98, f"rho_weat still collinear with rho_raw (corr={corr:.4f}); split failed"


# Recompute through the real φ* (gated on the HF cache)
def _real_phi() -> Arbiter:
    from board_generator.arbiter import load_consensus

    for arbiter in load_consensus(DEFAULT_CONSENSUS):
        if arbiter.ref == DEFAULT_CONSENSUS.primary:
            return arbiter
    raise RuntimeError("primary φ* not found")


@pytest.mark.integration
def test_real_fixture_regeneration_matches_and_provenance_holds() -> None:
    ref = DEFAULT_CONSENSUS.primary
    try:
        read_encoder_recipe(ref.model_id, ref.hf_revision)
    except FrameSnapshotError:
        pytest.skip(
            f"φ* snapshot not in local HF cache: {ref.model_id}@{ref.hf_revision}")

    sidecar = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    sidecar_axis = decode_f64_be_hex(
        sidecar["gender_axis"]["vector_f64_be_hex"])
    sidecar_mu = decode_f64_be_hex(sidecar["mu_bar"]["vector_f64_be_hex"])

    words = load_words(_RESOURCES / "words", _RESOURCES /
                       "frequencies" / "subtlex_us.csv").words
    attrs = read_attribute_words(
        _RESOURCES / "attribute_words" / "gender_attributes.csv")
    phi = _real_phi()

    # Fixture geometry is the frame's, computed through the raw pinned φ*.
    assert phi.ref == DEFAULT_CONSENSUS.primary
    rebuilt_axis = build_gender_axis(attrs, phi)
    reference = [a.word for a in attrs] + \
        [w.text for w in words if w.specification is not None]
    rebuilt_mu = build_mu_bar(reference, phi)
    # raw φ* axis == sidecar axis
    assert np.array_equal(rebuilt_axis, sidecar_axis)
    # the SINGLE μ̄, byte-identical
    assert np.array_equal(rebuilt_mu, sidecar_mu)

    board_id, card_texts = read_board_card_texts(_REFERENCE_BOARD)

    def regenerate() -> dict:
        return build_rho_reference(
            board_id=board_id, card_texts=card_texts, attributes=attrs, phi_star=phi,
            gender_axis=sidecar_axis, mu_bar=sidecar_mu, encoder=sidecar["encoder"],
            frame_id=sidecar["frame_id"],
        )

    first, second = regenerate(), regenerate()
    for a, b in zip(first["words"], second["words"], strict=True):
        for key in ("rho_raw", "rho_cent", "rho_weat"):
            # same process/model -> effectively exact
            assert abs(a[key] - b[key]) < 1e-12

    # The committed fixture equals a fresh regeneration (it is current).
    committed = _load_committed_fixture()
    assert committed["frame_id"] == sidecar["frame_id"]
    by_text = {w["text"]: w for w in committed["words"]}
    for row in first["words"]:
        ref_row = by_text[row["text"]]
        for key in ("rho_raw", "rho_cent", "rho_weat"):
            assert abs(row[key] - ref_row[key]) < 1e-9
