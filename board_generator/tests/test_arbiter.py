"""Offline tests for the arbiter embedding machinery, via injected stub encoders.

These NEVER touch the real backend: a HashEncoder supplies deterministic, network-free vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from board_generator.arbiter import Arbiter, ArbiterRef

from ._stub_encoders import HashEncoder


@pytest.fixture
def arbiter() -> Arbiter:
    return Arbiter(ref=ArbiterRef("stub/encoder", "rev-test"), encoder=HashEncoder())


def test_embed_is_deterministic(arbiter: Arbiter) -> None:
    assert np.array_equal(arbiter.embed("nurse"), arbiter.embed("nurse"))


def test_embed_lowercases(arbiter: Arbiter) -> None:
    assert np.array_equal(arbiter.embed("NURSE"), arbiter.embed("nurse"))


def test_embed_dtype_is_float64(arbiter: Arbiter) -> None:
    assert arbiter.embed("nurse").dtype == np.float64


def test_cos_self_similarity_is_one(arbiter: Arbiter) -> None:
    v = arbiter.embed("nurse")
    assert arbiter.cos(v, v) == pytest.approx(1.0)


def test_cos_is_symmetric(arbiter: Arbiter) -> None:
    a = arbiter.embed("nurse")
    b = arbiter.embed("hospital")
    assert arbiter.cos(a, b) == pytest.approx(arbiter.cos(b, a))


def test_cos_is_bounded(arbiter: Arbiter) -> None:
    words = ["nurse", "hospital", "dress", "engineer", "poetry"]
    vecs = [arbiter.embed(w) for w in words]
    for a in vecs:
        for b in vecs:
            assert -1.0 <= arbiter.cos(a, b) <= 1.0


def test_cos_zero_norm_returns_zero(arbiter: Arbiter) -> None:
    zero = np.zeros(16, dtype=np.float64)
    nonzero = arbiter.embed("nurse")
    assert arbiter.cos(zero, nonzero) == 0.0
    assert arbiter.cos(nonzero, zero) == 0.0
    assert arbiter.cos(zero, zero) == 0.0
