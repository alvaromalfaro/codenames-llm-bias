"""Placeholders for the invariants that need the generators."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Bank assembly (T=30, ~50/50 control/probe) not implemented")
def test_i1_bank_size_and_split() -> None: ...


@pytest.mark.skip(reason="Lexical composition (control=neutral, probe=valid dilemma) pending")
def test_i4_lexical_composition() -> None: ...


@pytest.mark.skip(reason="Role/gender independence test pending (see roles.py)")
def test_i5_role_gender_independence() -> None: ...


@pytest.mark.skip(reason="Eq. 4.1 consensus gate pending (verify_eq_4_1)")
def test_i6_dilemma_consensus() -> None: ...


@pytest.mark.skip(reason="Byte-for-byte reproducibility pending (full generation flow)")
def test_i8_reproducible_bank() -> None: ...
