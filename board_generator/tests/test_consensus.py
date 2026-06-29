"""Offline checks on the frozen DEFAULT_CONSENSUS trio - pure strings, NO network.

Stays in the default suite: it never loads a model, only inspects the committed pins and their
serialization. Guards the frozen instrument that makes Invariant I-8 hold.
"""

from __future__ import annotations

import re
import warnings

from board_generator.arbiter import DEFAULT_CONSENSUS, ConsensusSpec
from board_generator.board import arbiter_set_from_spec

# A pinned HF commit SHA: 40 lowercase hex chars (not a branch name like "main").
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_default_consensus_shape() -> None:
    assert DEFAULT_CONSENSUS is not None
    assert len(DEFAULT_CONSENSUS.encoders) == 3
    assert DEFAULT_CONSENSUS.primary.model_id == "sentence-transformers/all-mpnet-base-v2"
    assert DEFAULT_CONSENSUS.primary in DEFAULT_CONSENSUS.encoders


def test_default_consensus_pins_are_full_shas() -> None:
    for ref in DEFAULT_CONSENSUS.encoders:
        assert _FULL_SHA.match(ref.hf_revision), (
            f"{ref.model_id!r} is not pinned by a full 40-hex commit SHA: {ref.hf_revision!r}"
        )


def test_arbiter_set_round_trip() -> None:
    arbiters = arbiter_set_from_spec(DEFAULT_CONSENSUS)
    assert arbiters.primary in arbiters.consensus
    expected = [str(ref) for ref in DEFAULT_CONSENSUS.encoders]
    assert arbiters.consensus == expected
    for entry in arbiters.consensus:
        model_id, _, sha = entry.partition("@")
        assert model_id and _FULL_SHA.match(sha), f"entry not in model@sha form: {entry!r}"


def test_default_consensus_emits_no_same_lineage_warning() -> None:
    # The three are distinct families (all/gte/sentence), so re-building the spec must be silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning becomes an exception
        ConsensusSpec(
            encoders=DEFAULT_CONSENSUS.encoders,
            primary=DEFAULT_CONSENSUS.primary,
        )
