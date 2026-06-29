"""Read-only axis diagnostics for the gender-load diagnostic.

DIAGNOSTIC ONLY - ρ_w is reported construct-validity evidence, never an admission gate.

The ρ_w partition (load_filter.py) produced a negative τ_load on both specifications, with ρ_w
barely separating the male/female poles on bare-word φ*. Two very different causes look identical in
that report: either φ* embeddings share a large common offset (anisotropy drags every cosine toward
a common value, so the gender axis looks weak even when gender signal is present), or φ* bare-word
geometry simply carries little gender signal at all. This module tells them apart. The verdict
(below): mean-centering fixes ρ_w's sign/offset but not the separation, so the weakness is not mere
anisotropy - confirming ρ_w is unfit as a gate and is retained only as a reported diagnostic.

Two diagnostics, both per specification:

(1) WEAT-style effect size of the gender axis. Split the core words (weat_set != ()) into poles,
    compute ρ_w for each, and report mean ρ per pole, Cohen's d between the pole distributions, and
    a permutation p-value. This quantifies whether φ* bare-word separates male vs female at on
    the very WEAT cores that calibrate τ_load.

(2) Anisotropy test: ρ_w with vs without global mean-centering. μ̄ is the mean over every embedded
    item (attributes + loaded board words, deduplicated by text). φ̃(w) = φ*(w) - μ̄. ρ recomputed in
    the centered space, per word, alongside the raw ρ, plus the same effect-size block as (1) on
    ρ_centered so the two are directly comparable. If centering recovers the separation, the weak
    raw axis was anisotropy; if neither space separates, there is little gender signal.

Reuse, not reimplementation: build_gender_axis, rho and signed_load_toward_pole come straight from
load_filter.py. The raw and centered diagnostics use the same functions, just with two injected
arbiters over one shared embedding cache - raw_phi returns the cached φ* vector, centered_phi
returns that vector minus μ̄. A mean-difference axis is offset-invariant, so axis_centered equals
e_gen by construction; the cosine moves only because each point moves relative to the origin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from board_generator.arbiter import Arbiter, Encoder
from board_generator.balancing import _pooled_sd
from board_generator.lexicon import Specification, Word
from board_generator.load_filter import (
    AttributeWord,
    GenderPole,
    _word_pole,
    build_gender_axis,
    rho,
    signed_load_toward_pole,
)

# Number of label shuffles for the permutation p-value (the only free parameter here).
DEFAULT_PERMUTATIONS = 10000


@dataclass(frozen=True, slots=True)
class EffectSize:
    """Effect size of the gender axis on one pole split. Pure data - JSON-safe (no NaN/Inf).

    cohen_d and permutation_p are None when undefined (a pole with < 2 words gives an undefined
    pooled SD; an empty pole gives no permutation statistic), never NaN. n_permutations records how
    many label shuffles backed permutation_p.
    """

    n_male: int
    n_female: int
    mean_rho_male: float | None
    mean_rho_female: float | None
    cohen_d: float | None
    permutation_p: float | None
    n_permutations: int


@dataclass(frozen=True, slots=True)
class WordRow:
    """One word's load in both geometries: raw φ* and global-mean-centered φ̃.

    rho_* is the male-positive load cos(φ(w), axis); signed_load_* flips it toward the word's own
    pole (a strongly female word has signed_load > 0), exactly as in load_filter.py.
    """

    text: str
    pole: GenderPole
    rho_raw: float
    rho_centered: float
    signed_load_raw: float
    signed_load_centered: float


@dataclass(frozen=True, slots=True)
class SpecificationDiagnostics:
    """Per-specification diagnostics: both effect-size blocks plus the per-word rows.

    effect_raw is diagnostic (1) - the WEAT effect size on the CORE words in the raw space.
    effect_centered is the SAME block on the centered ρ, so the two are directly comparable. words
    covers core + expansion (raw and centered side by side).
    """

    specification: Specification
    effect_raw: EffectSize
    effect_centered: EffectSize
    words: list[WordRow]


@dataclass(frozen=True, slots=True)
class AxisDiagnostics:
    """Bank-level read-only diagnostic. Pure data - JSON-serializable (allow_nan=False holds).

    mu_bar_norm is ‖μ̄‖, a one-number summary of how large the shared offset is: a large value next
    to unit-axis cosines is the signature of anisotropy. n_embedded_items is the size of the
    deduplicated embedded set μ̄ averages over.
    """

    specifications: list[SpecificationDiagnostics]
    gender_axis_dim: int
    n_attributes: int
    n_embedded_items: int
    mu_bar_norm: float
    seed: int
    n_permutations: int
    arbiter_primary: str  # str(φ*.ref) - the single measurement encoder


class _CachingEncoder:
    """Memoizes the wrapped encoder by text, so each unique text is embedded exactly once.

    Arbiter.embed lowercases before calling encode, so the cache keys are the lowercased texts -
    i.e. the deduplicated embedded set μ̄ is built over. Satisfies the Encoder protocol.
    """

    def __init__(self, inner: Encoder) -> None:
        self._inner = inner
        self._cache: dict[str, NDArray[np.float64]] = {}

    def encode(self, text: str) -> NDArray[np.float64]:
        cached = self._cache.get(text)
        if cached is None:
            cached = np.asarray(self._inner.encode(text), dtype=np.float64)
            self._cache[text] = cached
        return cached

    @property
    def vectors(self) -> Mapping[str, NDArray[np.float64]]:
        """The deduplicated text -> raw embedding map accumulated so far."""
        return self._cache


class _CenteringEncoder:
    """Returns the cached raw embedding minus the global mean: φ̃(w) = φ*(w) - μ̄.

    Shares the raw _CachingEncoder's cache, so it triggers no new embedding work. Satisfies the
    Encoder protocol, which lets it be wrapped in an Arbiter and fed straight to build_gender_axis
    and rho.
    """

    def __init__(self, raw: _CachingEncoder, mu_bar: NDArray[np.float64]) -> None:
        self._raw = raw
        self._mu_bar = mu_bar

    def encode(self, text: str) -> NDArray[np.float64]:
        return self._raw.encode(text) - self._mu_bar


def build_axis_diagnostics(
    words: list[Word],
    attributes: list[AttributeWord],
    phi_star: Arbiter,
    *,
    seed: int,
    n_permutations: int = DEFAULT_PERMUTATIONS,
) -> AxisDiagnostics:
    """Assemble the read-only axis diagnostics. φ* is injected, nothing is written.

    DIAGNOSTIC ONLY - reports the gender-axis effect size and the raw-vs-centered ρ_w comparison as
    construct-validity evidence; it admits, rejects, or prunes no board word.

    Wraps φ* in a memoizing arbiter, embeds every attribute word and every loaded board word once,
    builds μ̄ over that deduplicated set, and derives a centering arbiter. The raw and centered
    gender axes and loads are then computed by REUSING build_gender_axis / rho on the two arbiters.
    A single seeded RNG, consumed in a fixed (specification, raw, centered) order, makes the
    permutation p-values deterministic.
    """
    raw_encoder = _CachingEncoder(phi_star.encoder)
    raw_phi = Arbiter(ref=phi_star.ref, encoder=raw_encoder)

    loaded = [w for w in words if w.specification is not None]
    for attribute in attributes:
        raw_phi.embed(attribute.word)
    for word in loaded:
        raw_phi.embed(word.text)

    cache = raw_encoder.vectors
    if not cache:
        raise ValueError(
            "no items to embed: need attribute words and loaded board words")
    mu_bar = np.mean(np.vstack(list(cache.values())), axis=0)
    centered_phi = Arbiter(
        ref=phi_star.ref, encoder=_CenteringEncoder(raw_encoder, mu_bar))

    e_gen = build_gender_axis(attributes, raw_phi)
    axis_centered = build_gender_axis(attributes, centered_phi)

    rng = np.random.default_rng(seed)
    spec_diagnostics: list[SpecificationDiagnostics] = []
    for specification in get_args(Specification):
        spec_words = sorted(
            (w for w in loaded if w.specification == specification), key=lambda w: w.text
        )
        rows = [_word_row(word, raw_phi, centered_phi, e_gen,
                          axis_centered) for word in spec_words]

        # The effect size uses only the grandfathered WEAT cores (weat_set != ()).
        core_texts = {w.text for w in spec_words if w.weat_set}
        core_rows = [row for row in rows if row.text in core_texts]
        # core split by pole, in the raw and centered spaces (same words, two geometries)
        male_raw = [row.rho_raw for row in core_rows if row.pole == "male"]
        female_raw = [row.rho_raw for row in core_rows if row.pole == "female"]
        male_centered = [
            row.rho_centered for row in core_rows if row.pole == "male"]
        female_centered = [
            row.rho_centered for row in core_rows if row.pole == "female"]

        effect_raw = _effect_size(male_raw, female_raw, rng, n_permutations)
        effect_centered = _effect_size(
            male_centered, female_centered, rng, n_permutations)
        spec_diagnostics.append(
            SpecificationDiagnostics(
                specification=specification,
                effect_raw=effect_raw,
                effect_centered=effect_centered,
                words=rows,
            )
        )

    return AxisDiagnostics(
        specifications=spec_diagnostics,
        gender_axis_dim=int(e_gen.shape[0]),
        n_attributes=len(attributes),
        n_embedded_items=len(cache),
        mu_bar_norm=float(np.linalg.norm(mu_bar)),
        seed=seed,
        n_permutations=n_permutations,
        arbiter_primary=str(phi_star.ref),
    )


def _word_row(
    word: Word,
    raw_phi: Arbiter,
    centered_phi: Arbiter,
    e_gen: NDArray[np.float64],
    axis_centered: NDArray[np.float64],
) -> WordRow:
    """Build one WordRow by reusing rho / signed_load_toward_pole in both geometries."""
    pole = _word_pole(word)
    rho_raw = rho(raw_phi, e_gen, word.text)
    rho_centered = rho(centered_phi, axis_centered, word.text)
    return WordRow(
        text=word.text,
        pole=pole,
        rho_raw=rho_raw,
        rho_centered=rho_centered,
        signed_load_raw=signed_load_toward_pole(rho_raw, pole),
        signed_load_centered=signed_load_toward_pole(rho_centered, pole),
    )


def _effect_size(
    male_rhos: list[float],
    female_rhos: list[float],
    rng: np.random.Generator,
    n_permutations: int,
) -> EffectSize:
    """Effect size of the gender axis between the two pole ρ distributions."""
    return EffectSize(
        n_male=len(male_rhos),
        n_female=len(female_rhos),
        mean_rho_male=float(np.mean(male_rhos)) if male_rhos else None,
        mean_rho_female=float(np.mean(female_rhos)) if female_rhos else None,
        cohen_d=_cohens_d(male_rhos, female_rhos),
        permutation_p=_permutation_p(
            male_rhos, female_rhos, rng, n_permutations),
        n_permutations=n_permutations,
    )


def _cohens_d(male: list[float], female: list[float]) -> float | None:
    """Cohen's d = (mean_male - mean_female) / pooled_SD; None when the pooled SD is undefined.

    Reuses balancing._pooled_sd (the simple sqrt((var_m + var_f)/2)) so the denominator matches the
    convention used elsewhere in the tool.
    """
    pooled = _pooled_sd(male, female)
    if pooled is None or pooled == 0.0:
        return None
    d = (float(np.mean(male)) - float(np.mean(female))) / pooled
    return d if np.isfinite(d) else None


def _permutation_p(
    male: list[float],
    female: list[float],
    rng: np.random.Generator,
    n_permutations: int,
) -> float | None:
    """Permutation p-value for |mean_male - mean_female|.

    Shuffle the pooled pole labels n times, recompute the absolute mean difference each time, and
    return the fraction of shuffles whose statistic is >= the observed one (plain fraction, no
    add-one). None when either pole is empty (no statistic to test).
    """
    if not male or not female:
        return None
    pooled = np.array(male + female, dtype=np.float64)
    n_male = len(male)
    observed = abs(float(np.mean(male)) - float(np.mean(female)))
    hits = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        diff = abs(
            float(np.mean(shuffled[:n_male])) - float(np.mean(shuffled[n_male:])))
        if diff >= observed:
            hits += 1
    return hits / n_permutations
