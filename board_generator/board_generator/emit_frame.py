"""Emit the versioned measurement-frame sidecar.

The measurement frame is a persisted, content-hashed record of the exact measurement instrument
(φ*'s frozen embedding recipe) plus the derived measurement geometry, the gender axis e_gen and
the centering mean μ̄.

This module only emits the sidecar. It reads φ*'s pinned recipe from the resolved Hugging Face
cache snapshot (never hard-coding pooling/normalize), rebuilds the axis and μ̄ through the SAME
existing generator functions generation uses (``build_gender_axis`` / ``build_mu_bar``), and writes
exactly one file: ``data/boards/measurement_frame.json``.

The frame is emitted via the RAW φ* only. It never constructs or routes through
``_CenteringEncoder`` (``axis_diagnostics``): that wrapper carries ``ref=phi_star.ref`` over a
modified geometry and would stamp a mislabelled identity into a persisted artifact. A
mean-difference is offset-invariant, so the raw axis equals the centered axis mathematically; we
still emit via raw φ*.

Canonical serialization + hash:
  * Vectors are encoded as lossless IEEE-754 float64 BIG-ENDIAN bytes, hex-joined
    (``vec.astype(">f8").tobytes().hex()``).
  * ``frame_id`` covers content only: it is ``sha256`` over the canonical JSON of the frame with
    ``frame_id``, ``created_at``, ``mu_bar.norm`` and ``mu_bar.cos_with_axis`` removed. So
    identity = {encoder recipe, both f64-be-hex vectors, construction, centering_reference}.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import try_to_load_from_cache
from numpy.typing import NDArray

from board_generator.board import DEFAULT_OUTPUT_DIR
from board_generator.lexicon import Word
from board_generator.load_filter import AttributeWord, build_gender_axis, build_mu_bar

SCHEMA_VERSION = 1
FRAME_FILENAME = "measurement_frame.json"

# Fields excluded from the content hash: identity is measurement content, not provenance.
_HASH_EXCLUDED_TOP = ("frame_id", "created_at")
_HASH_EXCLUDED_MU_BAR = ("norm", "cos_with_axis")

# sentence-transformers pooling flags -> the recipe label recorded in the frame.
_POOLING_FLAG_NAMES = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len",
    "pooling_mode_weightedmean_tokens": "weightedmean",
    "pooling_mode_lasttoken": "lasttoken",
}


class FrameSnapshotError(RuntimeError):
    """φ*'s pinned snapshot could not be resolved from the local HF cache (do not download)."""


# Encoder recipe: read from the resolved snapshot, never hard-coded
def recipe_from_snapshot_dir(
    snapshot_dir: Path, *, model_id: str, revision: str
) -> dict[str, Any]:
    """Read pooling + normalize from the checkpoint snapshot files (the instrument, not a constant).

    ``modules.json`` declares the module stack: ``normalize`` is true iff a ``…models.Normalize``
    module is present, and the Pooling module's ``path`` locates the pooling config whose active
    ``pooling_mode_*`` flag names the pooling. Raises ``FrameSnapshotError`` if a required file is
    absent — we reflect the instrument or fail loudly, never guess.
    """
    modules_path = snapshot_dir / "modules.json"
    if not modules_path.is_file():
        raise FrameSnapshotError(
            f"snapshot is missing modules.json: {modules_path}")
    modules = json.loads(modules_path.read_text(encoding="utf-8"))

    normalize = any(_module_type(m).rsplit(".", 1)
                    [-1] == "Normalize" for m in modules)

    pooling_module = next(
        (m for m in modules if _module_type(
            m).rsplit(".", 1)[-1] == "Pooling"), None
    )
    if pooling_module is None:
        raise FrameSnapshotError(
            f"snapshot declares no Pooling module: {modules_path}")
    pooling_cfg_path = snapshot_dir / \
        str(pooling_module.get("path", "")) / "config.json"
    if not pooling_cfg_path.is_file():
        raise FrameSnapshotError(
            f"snapshot is missing the pooling config: {pooling_cfg_path}")
    pooling = _pooling_label(json.loads(
        pooling_cfg_path.read_text(encoding="utf-8")))

    return {
        "name": model_id,
        "revision": revision,
        "pooling": pooling,
        "normalize": normalize,
    }


def read_encoder_recipe(
    model_id: str, revision: str, *, cache_dir: str | None = None
) -> dict[str, Any]:
    """Resolve φ*'s local HF-cache snapshot (no network) and read its recipe.

    Uses ``huggingface_hub.try_to_load_from_cache`` (honours ``HF_HOME``) to locate ``modules.json``
    for the pinned revision. If the snapshot is not already cached it returns a non-path sentinel;
    we raise ``FrameSnapshotError`` rather than download.
    """
    modules_path = try_to_load_from_cache(
        model_id, "modules.json", revision=revision, cache_dir=cache_dir
    )
    if not isinstance(modules_path, str):
        raise FrameSnapshotError(
            f"φ* snapshot not in the local HF cache for {model_id}@{revision} "
            f"(try_to_load_from_cache returned {modules_path!r}); refusing to download."
        )
    return recipe_from_snapshot_dir(
        Path(modules_path).parent, model_id=model_id, revision=revision
    )


def _module_type(module: dict[str, Any]) -> str:
    return str(module.get("type", ""))


def _pooling_label(config: dict[str, Any]) -> str:
    """Name the active pooling mode from the config's ``pooling_mode_*`` flags.

    Multiple concurrently-true modes (sentence-transformers can concatenate) are joined sorted with
    ``+`` so the label still reflects the instrument exactly. No true flag is a loud error.
    """
    active = sorted(
        name for flag, name in _POOLING_FLAG_NAMES.items() if config.get(flag) is True
    )
    if not active:
        raise FrameSnapshotError(
            f"pooling config has no active pooling_mode_* flag: {sorted(config)}"
        )
    return "+".join(active)


# Lossless float64 big-endian hex codec
def encode_f64_be_hex(vec: NDArray[np.float64]) -> str:
    """Encode a 1-D float64 vector as IEEE-754 big-endian bytes, hex-joined (lossless, exact)."""
    arr = np.asarray(vec)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {arr.shape}")
    if arr.dtype.kind != "f" or arr.dtype.itemsize != 8:
        raise ValueError(f"expected float64, got dtype {arr.dtype}")
    return arr.astype(">f8").tobytes().hex()


def decode_f64_be_hex(hex_str: str) -> NDArray[np.float64]:
    """Inverse of :func:`encode_f64_be_hex` — a native float64 array of the encoded values."""
    return np.frombuffer(bytes.fromhex(hex_str), dtype=">f8").astype(np.float64)


# Content hash + frame assembly
def frame_content_id(frame: dict[str, Any]) -> str:
    """sha256 over the canonical JSON of the frame's CONTENT subtree.

    Excludes ``frame_id`` + ``created_at`` (top level) and ``mu_bar.norm`` +
    ``mu_bar.cos_with_axis`` (diagnostics). Canonical form:
    ``json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=True)``.
    """
    pruned = copy.deepcopy(frame)
    for key in _HASH_EXCLUDED_TOP:
        pruned.pop(key, None)
    mu_bar = pruned.get("mu_bar")
    if isinstance(mu_bar, dict):
        for key in _HASH_EXCLUDED_MU_BAR:
            mu_bar.pop(key, None)
    payload = json.dumps(pruned, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assemble_frame(
    *,
    encoder: dict[str, Any],
    gender_axis: NDArray[np.float64],
    mu_bar: NDArray[np.float64],
    attribute_source: str,
    weat_sets: list[int],
    n_reference_items: int,
    attribute_counts_by_pole: dict[str, int],
    loaded_counts_by_pole: dict[str, int],
    generator_version: str,
    created_at: str,
) -> dict[str, Any]:
    """Build the sidecar from precomputed vectors + recipe, and stamp ``frame_id``.

    The diagnostic floats ``norm`` and ``cos_with_axis`` are derived deterministically from the same
    float64 vectors but sit OUTSIDE the hashed subtree.
    """
    frame: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": generator_version,
        "created_at": created_at,
        "encoder": {
            "name": encoder["name"],
            "revision": encoder["revision"],
            "pooling": encoder["pooling"],
            "normalize": encoder["normalize"],
        },
        "gender_axis": {
            "dim": int(gender_axis.shape[0]),
            "vector_f64_be_hex": encode_f64_be_hex(gender_axis),
            "construction": {
                "method": "mean_difference",
                "formula": "normalize(mean_male - mean_female)",
                "weat_sets": list(weat_sets),
                "attribute_source": attribute_source,
                "attribute_dedup_key": "(word, pole)",
            },
        },
        "mu_bar": {
            "dim": int(mu_bar.shape[0]),
            "vector_f64_be_hex": encode_f64_be_hex(mu_bar),
            "norm": float(np.linalg.norm(mu_bar)),
            "cos_with_axis": _cosine(mu_bar, gender_axis),
            "centering_reference": {
                "population": "attributes ∪ loaded(specification is not None)",
                "dedup_key": "lowercased text",
                "n_reference_items": int(n_reference_items),
                "attribute_counts_by_pole": {
                    "male": int(attribute_counts_by_pole.get("male", 0)),
                    "female": int(attribute_counts_by_pole.get("female", 0)),
                    "note": "post (word, pole) dedup",
                },
                "loaded_counts_by_pole": {
                    "male": int(loaded_counts_by_pole.get("male", 0)),
                    "female": int(loaded_counts_by_pole.get("female", 0)),
                    "note": "raw loaded pool pre-PSM; source of μ̄ off-axis component",
                },
            },
        },
    }
    frame["frame_id"] = frame_content_id(frame)
    return frame


def build_frame(
    phi_star: Any,
    words: list[Word],
    attributes: list[AttributeWord],
    *,
    attribute_source: str,
    encoder: dict[str, Any],
    generator_version: str,
    created_at: str,
) -> dict[str, Any]:
    """Rebuild axis + μ̄  via the existing generator functions and assemble the sidecar.

    Axis: raw ``build_gender_axis`` (no centering). μ̄: ``build_mu_bar`` over the same reference set
    ``build_sign_filter_report`` uses: every attribute word and every loaded board word
    (``specification is not None``), deduped by lowercased text. No seeded RNG is consumed.
    """
    gender_axis = build_gender_axis(attributes, phi_star)

    reference_texts = [attr.word for attr in attributes] + [
        w.text for w in words if w.specification is not None
    ]
    mu_bar = build_mu_bar(reference_texts, phi_star)
    n_reference_items = len({text.lower() for text in reference_texts})

    return assemble_frame(
        encoder=encoder,
        gender_axis=gender_axis,
        mu_bar=mu_bar,
        attribute_source=attribute_source,
        weat_sets=_weat_sets(attributes),
        n_reference_items=n_reference_items,
        attribute_counts_by_pole=_attribute_counts_by_pole(attributes),
        loaded_counts_by_pole=_loaded_counts_by_pole(words),
        generator_version=generator_version,
        created_at=created_at,
    )


def write_frame(frame: dict[str, Any], out_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write the frame to ``out_dir/measurement_frame.json``.

    Mirrors ``board.write_balance_report``: indent=2, ensure_ascii=False, LF-terminated. File
    formatting is independent of ``frame_id`` (hashed over the canonical compact form).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FRAME_FILENAME
    payload = json.dumps(frame, indent=2, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def _cosine(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _weat_sets(attributes: list[AttributeWord]) -> list[int]:
    """The WEAT set numbers present in the attributes (e.g. weat-6/7/8 -> [6, 7, 8])."""
    return sorted(
        {int(attr.weat_set.rsplit("-", 1)[-1])
         for attr in attributes if attr.weat_set}
    )


def _attribute_counts_by_pole(attributes: list[AttributeWord]) -> dict[str, int]:
    """Attribute counts per pole after (word, pole) dedup, the same dedup the axis applies."""
    pairs = {(attr.word, attr.gender_pole) for attr in attributes}
    counts = {"male": 0, "female": 0}
    for _word, pole in pairs:
        counts[pole] = counts.get(pole, 0) + 1
    return counts


def _loaded_counts_by_pole(words: list[Word]) -> dict[str, int]:
    """Per-pole counts of the LOADED reference words (``specification is not None``).

    A pure label count over each ``Word.gender_category``, without embedding or re-encoding. These are
    the majority of the μ̄ reference population and the reason μ̄  sits off the axis; neutral-labelled
    loaded words (if any) are counted under ``neutral`` so the breakdown stays exhaustive.
    """
    counts = {"male": 0, "female": 0, "neutral": 0}
    for word in words:
        if word.specification is not None:
            counts[word.gender_category] = counts.get(
                word.gender_category, 0) + 1
    return counts
