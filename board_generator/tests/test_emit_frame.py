"""Offline tests for the measurement-frame emitter (board_generator.emit_frame).

The hashing / encoding / determinism tests run without the real model, using synthetic 768-dim
geometry from the deterministic HashEncoder stub. The instrument invariants are shown by tests that
discriminate (perturb a float -> frame_id changes; a fake snapshot lacking Normalize -> recorded
normalize flips), never by argument. The one test that needs the true 768-dim embeddings is gated on
``integration`` (deselected by default).
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from board_generator.arbiter import DEFAULT_CONSENSUS, Arbiter, ArbiterRef
from board_generator.emit_frame import (
    FrameSnapshotError,
    assemble_frame,
    build_frame,
    decode_f64_be_hex,
    encode_f64_be_hex,
    frame_content_id,
    read_encoder_recipe,
    recipe_from_snapshot_dir,
)
from board_generator.lexicon import GenderCategory, Specification, Word
from board_generator.load_filter import (
    AttributeWord,
    build_gender_axis,
    build_mu_bar,
)

from ._stub_encoders import HashEncoder

# Fixtures / builders — synthetic 768-dim geometry, no model
DIM = 768

_ENCODER = {
    "name": "stub/phi-star",
    "revision": "rev-test",
    "pooling": "mean",
    "normalize": True,
}

_ATTRS = [
    AttributeWord(word="john", gender_pole="male",
                  source="weat", weat_set="weat-6"),
    AttributeWord(word="paul", gender_pole="male",
                  source="weat", weat_set="weat-7"),
    AttributeWord(word="mary", gender_pole="female",
                  source="weat", weat_set="weat-8"),
    AttributeWord(word="anna", gender_pole="female",
                  source="weat", weat_set="weat-6"),
]


def _phi() -> Arbiter:
    return Arbiter(ref=ArbiterRef("stub/phi-star", "rev-test"), encoder=HashEncoder(dim=DIM))


def _word(
    text: str,
    *,
    specification: Specification | None,
    gender: GenderCategory = "neutral",
) -> Word:
    return Word(
        text=text,
        gender_category=gender,
        word_kind="common",
        source="test",
        weat_set=(),
        dom_pos=None,
        ambiguous_pos=False,
        covariates={"subtlex_freq": float(
            len(text)), "length": float(len(text))},
        specification=specification,
    )


_WORDS = [
    _word("engineer", specification="gender-career", gender="male"),
    _word("nurse", specification="gender-science", gender="female"),
    # not in the reference set (specification is None)
    _word("banana", specification=None),
]


def _make_frame(*, created_at: str = "2026-07-18T00:00:00+00:00") -> dict:
    return build_frame(
        _phi(),
        _WORDS,
        _ATTRS,
        attribute_source="resources/attribute_words/gender_attributes.csv",
        encoder=_ENCODER,
        generator_version="0.1.0",
        created_at=created_at,
    )


# Determinism
def test_frame_id_and_vectors_deterministic() -> None:
    first = _make_frame()
    second = _make_frame()
    assert first["frame_id"] == second["frame_id"]
    assert (
        first["gender_axis"]["vector_f64_be_hex"] == second["gender_axis"]["vector_f64_be_hex"]
    )
    assert first["mu_bar"]["vector_f64_be_hex"] == second["mu_bar"]["vector_f64_be_hex"]


def test_frame_matches_direct_axis_and_mu_bar() -> None:
    """The emitted vectors are exactly the existing generator functions' RAW output."""
    frame = _make_frame()
    phi = _phi()
    axis = build_gender_axis(_ATTRS, phi)
    reference = [a.word for a in _ATTRS] + [
        w.text for w in _WORDS if w.specification is not None
    ]
    mu_bar = build_mu_bar(reference, phi)

    assert np.array_equal(decode_f64_be_hex(
        frame["gender_axis"]["vector_f64_be_hex"]), axis)
    assert np.array_equal(decode_f64_be_hex(
        frame["mu_bar"]["vector_f64_be_hex"]), mu_bar)
    assert frame["gender_axis"]["dim"] == DIM
    assert frame["mu_bar"]["dim"] == DIM


# Lossless codec round-trip
def test_codec_roundtrip_is_bit_exact() -> None:
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(DIM).astype(np.float64)
    restored = decode_f64_be_hex(encode_f64_be_hex(vec))
    assert restored.shape == (DIM,)
    assert restored.dtype == np.float64
    assert np.array_equal(restored, vec)  # bit-for-bit, not allclose


def test_encode_rejects_non_float64() -> None:
    with pytest.raises(ValueError):
        encode_f64_be_hex(np.zeros(DIM, dtype=np.float32))
    with pytest.raises(ValueError):
        encode_f64_be_hex(np.zeros((2, DIM), dtype=np.float64))


# frame_id sensitivity, the recipe and vectors are inside the hash
def _assemble(axis: np.ndarray, mu: np.ndarray, encoder: dict) -> dict:
    return assemble_frame(
        encoder=encoder,
        gender_axis=axis,
        mu_bar=mu,
        attribute_source="src.csv",
        weat_sets=[6, 7, 8],
        n_reference_items=5,
        attribute_counts_by_pole={"male": 2, "female": 2},
        loaded_counts_by_pole={"male": 1, "female": 1},
        generator_version="0.1.0",
        created_at="2026-07-18T00:00:00+00:00",
    )


def test_frame_id_changes_when_one_axis_float_perturbed() -> None:
    rng = np.random.default_rng(1)
    axis = rng.standard_normal(DIM).astype(np.float64)
    mu = rng.standard_normal(DIM).astype(np.float64)
    base = _assemble(axis, mu, _ENCODER)

    perturbed_axis = axis.copy()
    perturbed_axis[0] = np.nextafter(
        perturbed_axis[0], np.inf)  # smallest possible change
    perturbed = _assemble(perturbed_axis, mu, _ENCODER)

    assert base["frame_id"] != perturbed["frame_id"]


def test_frame_id_changes_when_normalize_flag_flipped() -> None:
    rng = np.random.default_rng(2)
    axis = rng.standard_normal(DIM).astype(np.float64)
    mu = rng.standard_normal(DIM).astype(np.float64)
    base = _assemble(axis, mu, {**_ENCODER, "normalize": True})
    flipped = _assemble(axis, mu, {**_ENCODER, "normalize": False})
    assert base["frame_id"] != flipped["frame_id"]


def test_frame_id_ignores_created_at_and_diagnostic_floats() -> None:
    """Content-only identity: created_at, mu_bar.norm, mu_bar.cos_with_axis are OUTSIDE the hash."""
    frame = _make_frame(created_at="2026-07-18T00:00:00+00:00")
    other = _make_frame(created_at="1999-01-01T12:34:56+00:00")
    assert frame["frame_id"] == other["frame_id"]

    # garbage the excluded fields -> frame_content_id is unchanged.
    tampered = copy.deepcopy(frame)
    tampered["created_at"] = "tampered"
    tampered["mu_bar"]["norm"] = -12345.0
    tampered["mu_bar"]["cos_with_axis"] = 999.0
    assert frame_content_id(tampered) == frame["frame_id"]


# Schema shape + derived metadata
def test_frame_shape_and_derived_metadata() -> None:
    frame = _make_frame()
    assert frame["schema_version"] == 1
    assert frame["encoder"] == _ENCODER
    assert frame["gender_axis"]["construction"]["weat_sets"] == [6, 7, 8]
    assert frame["gender_axis"]["construction"]["method"] == "mean_difference"
    ref = frame["mu_bar"]["centering_reference"]
    # attributes(4 unique) U loaded(engineer, nurse), banana has specification None, excluded.
    assert ref["n_reference_items"] == 6
    assert ref["attribute_counts_by_pole"]["male"] == 2
    assert ref["attribute_counts_by_pole"]["female"] == 2
    # loaded pole counts come from Word.gender_category (engineer=male, nurse=female).
    assert ref["loaded_counts_by_pole"]["male"] == 1
    assert ref["loaded_counts_by_pole"]["female"] == 1
    # cos with the axis is a finite diagnostic float
    assert isinstance(frame["mu_bar"]["cos_with_axis"], float)


# Snapshot-read, reflects the instrument, not constants
_MEAN_POOLING = {"pooling_mode_mean_tokens": True}
_CLS_POOLING = {"pooling_mode_cls_token": True}
_ST = "sentence_transformers.models"


def _write_snapshot(tmp_path, *, include_normalize: bool, pooling_flags: dict) -> None:
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": f"{_ST}.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": f"{_ST}.Pooling"},
    ]
    if include_normalize:
        modules.append(
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": f"{_ST}.Normalize"}
        )
    (tmp_path / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    pooling_dir = tmp_path / "1_Pooling"
    pooling_dir.mkdir()
    base_flags = {
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
    }
    base_flags.update(pooling_flags)
    (pooling_dir / "config.json").write_text(json.dumps(base_flags), encoding="utf-8")


def test_recipe_reads_normalize_true_and_mean_pooling(tmp_path) -> None:
    _write_snapshot(tmp_path, include_normalize=True,
                    pooling_flags=_MEAN_POOLING)
    recipe = recipe_from_snapshot_dir(tmp_path, model_id="m", revision="r")
    assert recipe == {"name": "m", "revision": "r",
                      "pooling": "mean", "normalize": True}


def test_recipe_records_normalize_false_when_module_absent(tmp_path) -> None:
    """Drop the Normalize module -> the emitter records normalize=false."""
    _write_snapshot(tmp_path, include_normalize=False,
                    pooling_flags=_MEAN_POOLING)
    recipe = recipe_from_snapshot_dir(tmp_path, model_id="m", revision="r")
    assert recipe["normalize"] is False


def test_recipe_pooling_reflects_config(tmp_path) -> None:
    """A cls-pooling config -> the emitter records pooling='cls', not 'mean'."""
    _write_snapshot(tmp_path, include_normalize=True,
                    pooling_flags=_CLS_POOLING)
    recipe = recipe_from_snapshot_dir(tmp_path, model_id="m", revision="r")
    assert recipe["pooling"] == "cls"


def test_recipe_raises_when_no_active_pooling_flag(tmp_path) -> None:
    _write_snapshot(tmp_path, include_normalize=True, pooling_flags={})
    with pytest.raises(FrameSnapshotError):
        recipe_from_snapshot_dir(tmp_path, model_id="m", revision="r")


def test_read_encoder_recipe_resolves_via_cache(tmp_path, monkeypatch) -> None:
    _write_snapshot(tmp_path, include_normalize=True,
                    pooling_flags=_MEAN_POOLING)
    monkeypatch.setattr(
        "board_generator.emit_frame.try_to_load_from_cache",
        lambda *a, **k: str(tmp_path / "modules.json"),
    )
    recipe = read_encoder_recipe("m", "r")
    assert recipe["pooling"] == "mean"
    assert recipe["normalize"] is True


def test_read_encoder_recipe_raises_when_not_cached(monkeypatch) -> None:
    monkeypatch.setattr(
        "board_generator.emit_frame.try_to_load_from_cache", lambda *a, **k: None
    )
    with pytest.raises(FrameSnapshotError):
        read_encoder_recipe("m", "r")


# Reads the REAL pinned φ* snapshot recipe (gated on the HF cache)
@pytest.mark.integration
def test_real_phi_star_recipe_is_mean_and_normalized() -> None:
    ref = DEFAULT_CONSENSUS.primary
    try:
        recipe = read_encoder_recipe(ref.model_id, ref.hf_revision)
    except FrameSnapshotError:
        pytest.skip(
            f"φ* snapshot not in local HF cache: {ref.model_id}@{ref.hf_revision}")
    assert recipe["pooling"] == "mean"
    assert recipe["normalize"] is True
