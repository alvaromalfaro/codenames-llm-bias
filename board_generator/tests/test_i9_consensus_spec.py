"""φ* is a member of the consensus set; plus the other ConsensusSpec guards."""

from __future__ import annotations

import pytest

from board_generator.arbiter import ArbiterRef, ConsensusSpec
from board_generator.board import arbiter_set_from_spec


def test_i9_primary_in_consensus(valid_consensus_spec: ConsensusSpec) -> None:
    assert valid_consensus_spec.primary in valid_consensus_spec.encoders


def test_i9_serialized_primary_in_consensus(valid_consensus_spec: ConsensusSpec) -> None:
    arbiters = arbiter_set_from_spec(valid_consensus_spec)
    assert arbiters.primary in arbiters.consensus


def test_i9_rejects_primary_outside_consensus() -> None:
    ref1 = ArbiterRef("sentence-transformers/all-mpnet-base-v2", "rev-a")
    ref2 = ArbiterRef("Alibaba-NLP/gte-large-en-v1.5", "rev-b")
    outsider = ArbiterRef("nomic-ai/nomic-embed-text-v1.5", "rev-c")
    with pytest.raises(ValueError, match="primary"):
        ConsensusSpec(encoders=(ref1, ref2), primary=outsider)


def test_rejects_empty_consensus() -> None:
    ref = ArbiterRef("sentence-transformers/all-mpnet-base-v2", "rev-a")
    with pytest.raises(ValueError, match="empty"):
        ConsensusSpec(encoders=(), primary=ref)


def test_rejects_missing_revision() -> None:
    ref = ArbiterRef("sentence-transformers/all-mpnet-base-v2", "")
    with pytest.raises(ValueError, match="HF revision"):
        ConsensusSpec(encoders=(ref,), primary=ref)


def test_warns_on_same_lineage_siblings() -> None:
    sibling_a = ArbiterRef("sentence-transformers/all-mpnet-base-v2", "rev-a")
    sibling_b = ArbiterRef("sentence-transformers/all-MiniLM-L6-v2", "rev-b")
    with pytest.warns(UserWarning, match="same-lineage"):
        ConsensusSpec(encoders=(sibling_a, sibling_b), primary=sibling_a)
