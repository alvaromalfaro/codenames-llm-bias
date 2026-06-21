"""The ONE test that exercises the real sentence-transformers backend.

Deselected by default (`-m 'not integration'`) and self-skips offline: it requires both the
sentence_transformers package AND the RUN_ARBITER_INTEGRATION env var (so it never downloads a
model unless explicitly opted in). The operator supplies the pinned (model id, HF revision) via
env vars - the test never fabricates a revision. This is the only test allowed to touch the real
backend.
"""

from __future__ import annotations

import os

import pytest

from board_generator.arbiter import ArbiterRef, ConsensusSpec, load_consensus


@pytest.mark.integration
def test_real_backend_embeds_and_scores() -> None:
    pytest.importorskip("sentence_transformers")
    if not os.environ.get("RUN_ARBITER_INTEGRATION"):
        pytest.skip("set RUN_ARBITER_INTEGRATION to run the real-backend arbiter test")

    model_id = os.environ.get("ARBITER_MODEL_ID")
    hf_revision = os.environ.get("ARBITER_HF_REVISION")
    if not model_id or not hf_revision:
        pytest.skip("set ARBITER_MODEL_ID and ARBITER_HF_REVISION (pinned) to run this test")

    primary = ArbiterRef(model_id, hf_revision)
    spec = ConsensusSpec(encoders=(primary,), primary=primary)

    arbiters = load_consensus(spec)
    assert len(arbiters) == 1

    arbiter = arbiters[0]
    nurse = arbiter.embed("nurse")
    hospital = arbiter.embed("hospital")

    assert arbiter.cos(nurse, nurse) == pytest.approx(1.0, abs=1e-6)
    assert -1.0 <= arbiter.cos(nurse, hospital) <= 1.0
