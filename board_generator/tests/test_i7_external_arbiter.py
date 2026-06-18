"""No consensus arbiter is one of the evaluated models."""

from __future__ import annotations

import pytest

from board_generator.arbiter import ArbiterRef, ConsensusSpec, assert_external


def test_i7_accepts_external_consensus(valid_consensus_spec: ConsensusSpec) -> None:
    assert_external(valid_consensus_spec)  # external arbiters -> no raise


def test_i7_rejects_evaluated_model() -> None:
    evaluated = ArbiterRef("meta-llama/Llama-3.1-8B-Instruct", "rev-x")
    external = ArbiterRef("sentence-transformers/all-mpnet-base-v2", "rev-y")
    with pytest.raises(ValueError, match="evaluated model"):
        ConsensusSpec(encoders=(evaluated, external), primary=external)
