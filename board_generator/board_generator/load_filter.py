"""Gender-load diagnostic (not an admission gate).

DIAGNOSTIC ONLY - ρ_w is reported construct-validity evidence, never an admission gate. Board
inclusion of expansion words is by the source-disparity criterion; the WEAT core is grandfathered.
Nothing in this module admits, rejects, or prunes a board word: the `admitted` / `survivors` fields
below are a reported diagnostic partition, never enforced on the pool. (ρ_w on bare-word φ* was
tested as a gate and failed - a moderate, non-significant effect; per-word divergence from the
labels; it broke the covariate balance - so it was demoted to a diagnostic).

It computes the signed gender load ρ_w = cos(φ*(w), e_gen), reusing the same primary arbiter φ* and
gender axis the downstream metrics use, here as a measurement instrument - not the dilemma consensus
gate. ρ_w is a continuous measure, so it uses a single fixed encoder (φ*), never a consensus average
(geometries from different encoders are not commensurable).

The WEAT core (weat_set != ()) is canonical and citable; only expansion words (weat_set == ()) are
scored against τ_load, and the core is exactly what calibrates the per-specification threshold. The
resulting partition (which expansion words clear τ_load) is a reported diagnostic lens, not a filter
applied to the board.

The axis is male-minus-female, so male is positive by construction and there is no PCA sign
ambiguity. Given a fixed φ*, the axis, τ_load and the partition are deterministic.

Every public function takes the arbiter φ* injected (mirroring dilemma.py) and reads nothing itself,
except the thin read_attribute_words CSV helper. It is therefore fully offline-testable with
stub-backed Arbiters.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from numpy.typing import NDArray

from board_generator.arbiter import Arbiter
from board_generator.balancing import BalanceReport, run_balancing
from board_generator.lexicon import Specification, Word

# Gender pole of an attribute word: the axis is male - female (male positive).
GenderPole = Literal["male", "female"]

_VALID_POLE: frozenset[str] = frozenset(get_args(GenderPole))

# The only documented free parameter: the per-spec core quantile anchoring τ_load.
DEFAULT_QUANTILE = 0.10


@dataclass(frozen=True, slots=True)
class AttributeWord:
    """One gender-attribute row (resources/attribute_words/gender_attributes.csv).

    Schema word,gender_pole,source,weat_set - different from the board word CSVs and read directly,
    not via lexicon.load_words. These are the names/pronouns/kinship terms that define e_gen.
    """

    word: str  # lowercased
    gender_pole: GenderPole
    source: str
    weat_set: str


@dataclass(frozen=True, slots=True)
class WordLoad:
    """The load verdict for one word: its signed load toward its own pole and whether it survives.

    rho is the raw male-positive load cos(φ*(w), e_gen); signed_load is rho flipped to point toward
    the word's own pole (so a strongly female word has signed_load > 0). grandfathered core words
    are always admitted; expansion words are admitted iff signed_load >= τ_load.
    """

    text: str
    pole: GenderPole
    rho: float
    signed_load: float
    grandfathered: bool
    admitted: bool


@dataclass(frozen=True, slots=True)
class SpecificationLoadFilter:
    """Per-specification load-filter outcome: τ_load, the per-word verdicts and the survivors."""

    specification: Specification
    tau_load: float
    quantile: float
    # grandfathered calibrators (weat_set != ()), always admitted
    core: list[WordLoad]
    expansion: list[WordLoad]  # filtered candidates (weat_set == ())
    survivors: list[str]  # texts kept = all core + admitted expansion
    n_core: int
    n_expansion_admitted: int
    n_expansion_rejected: int


@dataclass(frozen=True, slots=True)
class LoadFilterReport:
    """Non-destructive diagnostic. Pure data - JSON-serializable (allow_nan=False holds)."""

    specifications: list[SpecificationLoadFilter]
    gender_axis_dim: int
    n_attributes: int
    seed: int
    quantile: float
    arbiter_primary: str  # str(φ*.ref) - the single measurement encoder
    rebalance: BalanceReport


@dataclass(frozen=True, slots=True)
class WordSignLoad:
    """The sign-criterion verdict for one word on the global-mean-CENTERED axis.

    rho_centered is the male-positive load cos(φ*(w) - μ̄, axis_centered); signed_load_centered flips
    it toward the word's own pole (a strongly female word has signed_load_centered > 0). Core words
    are always admitted; an expansion word is admitted iff signed_load_centered > δ (landing
    strictly on the correct side of the centered axis, with an a-priori margin δ).
    """

    text: str
    pole: GenderPole
    rho_centered: float
    signed_load_centered: float
    grandfathered: bool
    admitted: bool


@dataclass(frozen=True, slots=True)
class SpecificationSignFilter:
    """Per-specification sign-criterion outcome at one δ: per-word verdicts and the survivors."""

    specification: Specification
    delta: float
    # grandfathered calibrators (weat_set != ()), always admitted
    core: list[WordSignLoad]
    expansion: list[WordSignLoad]  # filtered candidates (weat_set == ())
    survivors: list[str]  # texts kept = all core + admitted expansion
    n_core: int
    n_expansion_admitted: int
    n_expansion_rejected: int


@dataclass(frozen=True, slots=True)
class SignFilterReport:
    """Non-destructive sign-criterion diagnostic at one δ. Pure data - JSON-safe (allow_nan=False).

    A second, additive specification alongside the quantile LoadFilterReport: it admits expansion
    words by the CENTERED sign rule instead of the raw core-quantile τ_load. mu_bar_norm = ‖μ̄‖ is
    the global mean used to center, computed once over n_reference_items deduplicated texts.
    """

    specifications: list[SpecificationSignFilter]
    delta: float
    gender_axis_dim: int
    n_attributes: int
    n_reference_items: int
    mu_bar_norm: float
    seed: int
    arbiter_primary: str  # str(φ*.ref) - the single measurement encoder
    rebalance: BalanceReport


def read_attribute_words(path: Path) -> list[AttributeWord]:
    """Read the gender-attribute CSV (word,gender_pole,source,weat_set) into AttributeWords.

    Words are lowercased (φ*.embed lowercases too). Blank words are skipped; an unknown gender_pole
    is a hard error.
    """
    attributes: list[AttributeWord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        # +1 header, +1 to 1-based
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            word = (row["word"] or "").strip().lower()
            if not word:
                continue
            pole = (row["gender_pole"] or "").strip()
            if pole not in _VALID_POLE:
                raise ValueError(
                    f"{path.name}:{line_no}: unknown gender_pole {pole!r} for {word!r}"
                )
            gender_pole: GenderPole = pole  # type: ignore[assignment]
            attributes.append(
                AttributeWord(
                    word=word,
                    gender_pole=gender_pole,
                    source=(row["source"] or "").strip(),
                    weat_set=(row["weat_set"] or "").strip(),
                )
            )
    return attributes


def build_mu_bar(texts: list[str], phi_star: Arbiter) -> NDArray[np.float64]:
    """Mean of the bare-word φ* embeddings over texts, deduplicated by lowercased text.

    DIAGNOSTIC ONLY - μ̄ centers the geometry for the reported centered-ρ diagnostic; it gates
    nothing. This is the single global mean μ̄ used to center the space: the caller computes it once
    over the full reference set (every attribute word U every loaded board word) and reuses that μ̄
    for the centered axis and every centered ρ, for both specifications - so sign comparisons stay
    commensurable and the bank stays reproducible. Dedup is by lowercased text (φ*.embed lowercases
    too), and the unique texts are sorted, so a duplicated text never shifts μ̄ and the result is
    deterministic. Raises on an empty reference set (nothing to average).
    """
    unique = sorted({text.lower() for text in texts})
    if not unique:
        raise ValueError("cannot build μ̄: the reference set is empty")
    return np.mean(np.vstack([phi_star.embed(text) for text in unique]), axis=0)


def build_gender_axis(
    attributes: list[AttributeWord],
    phi_star: Arbiter,
    *,
    centered: bool = False,
    mu_bar: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Build the gender axis e_gen by mean-difference (DIAGNOSTIC instrument, not a gate).

        e_gen = normalize(mean_{w in male} φ*(w) - mean_{w in female} φ*(w))

    Uses all attributes (names + pronouns + kinship). Deduplicates to unique (lowercased word, pole)
    pairs first, so repeated attributes (e.g. BROTHER/SON/HE shared by weat-7 & weat-8) are not
    double-weighted. Raises if a word appears under conflicting poles. Male is positive by
    construction (axis is male - female).

    When centered (additive; default False preserves the raw axis byte for byte), the pole means are
    taken on the centered geometry φ̃(x) = φ*(x) - μ̄:

        axis_centered = normalize((mean_male(φ̃)) - (mean_female(φ̃)))

    mu_bar is then required (raises otherwise). A mean-difference is offset-invariant, so the
    centered axis equals the raw axis; the centering is implemented faithfully and explicitly so the
    centered ρ (which is not offset-invariant) is computed against the same construction.
    """
    if centered and mu_bar is None:
        raise ValueError("centered gender axis requires mu_bar")

    pole_by_word: dict[str, GenderPole] = {}
    for attr in attributes:
        existing = pole_by_word.get(attr.word)
        if existing is not None and existing != attr.gender_pole:
            raise ValueError(
                f"attribute {attr.word!r} appears under conflicting poles: "
                f"{existing!r} vs {attr.gender_pole!r}"
            )
        pole_by_word[attr.word] = attr.gender_pole

    male = sorted(w for w, pole in pole_by_word.items() if pole == "male")
    female = sorted(w for w, pole in pole_by_word.items() if pole == "female")
    if not male or not female:
        raise ValueError(
            "gender axis needs at least one male and one female attribute")

    mean_male = _mean_embedding(male, phi_star)
    mean_female = _mean_embedding(female, phi_star)
    if centered:
        assert mu_bar is not None  # narrowed above; explicit for type-checkers
        mean_male = mean_male - mu_bar
        mean_female = mean_female - mu_bar
    return _normalize(mean_male - mean_female)


def rho(
    phi_star: Arbiter,
    e_gen: NDArray[np.float64],
    text: str,
    *,
    centered: bool = False,
    mu_bar: NDArray[np.float64] | None = None,
) -> float:
    """Signed gender load: ρ_w = cos(φ*(text), e_gen). Male is positive by construction.

    DIAGNOSTIC ONLY - ρ_w is reported construct-validity evidence, never an admission gate. When
    centered (additive; default False preserves the raw load), the load is taken on the centered
    geometry: ρ_centered(text) = cos(φ*(text) − μ̄, e_gen). mu_bar is then required (raises
    otherwise) and e_gen is the centered axis. Unlike the axis, ρ is not offset-invariant, so
    centering shifts it - that is precisely the sign/offset fix this mode adds.
    """
    if centered and mu_bar is None:
        raise ValueError("centered rho requires mu_bar")
    vec = phi_star.embed(text)
    if centered:
        assert mu_bar is not None  # narrowed above; explicit for type-checkers
        vec = vec - mu_bar
    return phi_star.cos(vec, e_gen)


def signed_load_toward_pole(rho_value: float, pole: GenderPole) -> float:
    """Flip the male-positive load to point toward the word's own pole (female words flip sign)."""
    return rho_value if pole == "male" else -rho_value


def calibrate_tau_load(
    core_words: list[Word], phi_star: Arbiter, e_gen: NDArray[np.float64], quantile: float
) -> float:
    """Calibrate the diagnostic threshold τ_load(s) from the per-spec WEAT core.

    DIAGNOSTIC ONLY - τ_load is a reported reference level for the ρ_w diagnostic, not an admission
    gate; no board word is included or excluded by it.

    τ_load(s) = quantile_p({signed-load-toward-its-own-pole(w) : w in core_s}) with p = quantile
    (the single documented free parameter, default 0.10), linear interpolation. Core words are
    never filtered - they are exactly what calibrates the threshold. Raises on an empty core
    (nothing to calibrate against).
    """
    if not core_words:
        raise ValueError(
            "cannot calibrate τ_load: the specification has no WEAT core words")
    loads = [
        signed_load_toward_pole(rho(phi_star, e_gen, w.text), _word_pole(w)) for w in core_words
    ]
    return float(np.quantile(loads, quantile, method="linear"))


def filter_expansion(
    expansion_words: list[Word], tau: float, phi_star: Arbiter, e_gen: NDArray[np.float64]
) -> list[WordLoad]:
    """Label expansion words by τ_load (diagnostic partition, not an admission gate).

    DIAGNOSTIC ONLY - this labels each expansion word, it does not remove it from the board.
    Inclusion is by the source criterion. An expansion word of pole P is labeled admitted iff
    signed-load-toward-P(w) >= τ (non-strict: an exact tie passes), else admitted=False; this is the
    reported partition (topically-correct but gender-weak bridges land below τ), never a filter
    applied to the pool.
    """
    verdicts: list[WordLoad] = []
    for word in expansion_words:
        pole = _word_pole(word)
        rho_value = rho(phi_star, e_gen, word.text)
        signed = signed_load_toward_pole(rho_value, pole)
        verdicts.append(
            WordLoad(
                text=word.text,
                pole=pole,
                rho=rho_value,
                signed_load=signed,
                grandfathered=False,
                admitted=signed >= tau,
            )
        )
    return verdicts


def filter_expansion_sign(
    expansion_words: list[Word],
    delta: float,
    phi_star: Arbiter,
    axis_centered: NDArray[np.float64],
    mu_bar: NDArray[np.float64],
) -> list[WordSignLoad]:
    """Label expansion words by the centered sign criterion (diagnostic partition).

    DIAGNOSTIC ONLY - a second reported lens alongside the raw τ_load partition; like it, it labels
    words and removes nothing. An expansion word of pole P is labeled admitted iff its
    signed-load-toward-P on the centered geometry is strictly above the a-priori margin δ
    (signed_load_centered > δ): it lands strictly on the correct side of the centered axis. δ = 0
    means "strictly correct side"; an exact tie at δ is admitted=False (strict '>'). Centering fixes
    the sign/offset of the raw poles (which overlap), but adds no separation.
    """
    verdicts: list[WordSignLoad] = []
    for word in expansion_words:
        pole = _word_pole(word)
        rho_value = rho(phi_star, axis_centered, word.text,
                        centered=True, mu_bar=mu_bar)
        signed = signed_load_toward_pole(rho_value, pole)
        verdicts.append(
            WordSignLoad(
                text=word.text,
                pole=pole,
                rho_centered=rho_value,
                signed_load_centered=signed,
                grandfathered=False,
                admitted=signed > delta,
            )
        )
    return verdicts


def build_load_filter_report(
    words: list[Word],
    attributes: list[AttributeWord],
    phi_star: Arbiter,
    *,
    seed: int,
    quantile: float = DEFAULT_QUANTILE,
) -> LoadFilterReport:
    """Assemble the diagnostic report. φ* is injected.

    DIAGNOSTIC ONLY - this reports the ρ_w partition; it admits/prunes no board word. survivors is
    the reported "all core + diagnostically-admitted expansion" set, not a pruned pool. The embedded
    re-balance shows what the partition would do - empirically it breaks balance, which is why the
    partition is never applied.

    Builds e_gen once, then per specification splits the loaded words into grandfathered core
    (weat_set != ()) and diagnostically-labelled expansion (weat_set == ()), calibrates τ_load(s) on
    the core, and labels the expansion. Finally re-runs balancing.run_balancing over the reported
    survivors of both specifications and embeds its report.
    """
    e_gen = build_gender_axis(attributes, phi_star)

    spec_filters: list[SpecificationLoadFilter] = []
    survivor_words: list[Word] = []
    for specification in get_args(Specification):
        loaded = [w for w in words if w.specification == specification]
        core_words = sorted(
            (w for w in loaded if w.weat_set), key=lambda w: w.text)
        expansion_words = sorted(
            (w for w in loaded if not w.weat_set), key=lambda w: w.text)

        tau = calibrate_tau_load(core_words, phi_star, e_gen, quantile)
        core_loads = [
            WordLoad(
                text=w.text,
                pole=_word_pole(w),
                rho=(rho_value := rho(phi_star, e_gen, w.text)),
                signed_load=signed_load_toward_pole(rho_value, _word_pole(w)),
                grandfathered=True,
                admitted=True,
            )
            for w in core_words
        ]
        expansion_loads = filter_expansion(
            expansion_words, tau, phi_star, e_gen)

        survivors = [load.text for load in core_loads] + [
            load.text for load in expansion_loads if load.admitted
        ]
        survivor_set = set(survivors)
        survivor_words.extend(w for w in loaded if w.text in survivor_set)

        admitted = sum(1 for load in expansion_loads if load.admitted)
        spec_filters.append(
            SpecificationLoadFilter(
                specification=specification,
                tau_load=tau,
                quantile=quantile,
                core=core_loads,
                expansion=expansion_loads,
                survivors=survivors,
                n_core=len(core_loads),
                n_expansion_admitted=admitted,
                n_expansion_rejected=len(expansion_loads) - admitted,
            )
        )

    rebalance = run_balancing(survivor_words, seed=seed).report
    return LoadFilterReport(
        specifications=spec_filters,
        gender_axis_dim=int(e_gen.shape[0]),
        n_attributes=len(attributes),
        seed=seed,
        quantile=quantile,
        arbiter_primary=str(phi_star.ref),
        rebalance=rebalance,
    )


def build_sign_filter_report(
    words: list[Word],
    attributes: list[AttributeWord],
    phi_star: Arbiter,
    *,
    seed: int,
    delta: float,
) -> SignFilterReport:
    """Assemble the sign-criterion diagnostic report at one δ. φ* is injected.

    DIAGNOSTIC ONLY - the centered-sign analogue of build_load_filter_report; it REPORTS the
    partition and prunes nothing. survivors is the reported set, not a pruned pool.

    Builds μ̄ once via build_mu_bar over the full reference set = every attribute word U every
    loaded board word (w.specification is not None), deduplicated by lowercased text - the SAME set
    axis_diagnostics averages over, so μ̄ is comparable. The same μ̄ centers the axis and every ρ,
    for both specifications. Then, per specification, grandfathers the WEAT core (weat_set != (),
    always labelled admitted) and labels expansion (weat_set == ()) admitted iff
    signed_load_centered > δ (strict). Finally re-runs balancing.run_balancing over the reported
    survivors of both specifications and embeds its report.
    """
    loaded_all = [w for w in words if w.specification is not None]
    reference_texts = [attr.word for attr in attributes] + \
        [w.text for w in loaded_all]
    mu_bar = build_mu_bar(reference_texts, phi_star)
    n_reference_items = len({text.lower() for text in reference_texts})
    axis_centered = build_gender_axis(
        attributes, phi_star, centered=True, mu_bar=mu_bar)

    spec_filters: list[SpecificationSignFilter] = []
    survivor_words: list[Word] = []
    for specification in get_args(Specification):
        loaded = [w for w in words if w.specification == specification]
        core_words = sorted(
            (w for w in loaded if w.weat_set), key=lambda w: w.text)
        expansion_words = sorted(
            (w for w in loaded if not w.weat_set), key=lambda w: w.text)

        core_loads = [
            WordSignLoad(
                text=w.text,
                pole=_word_pole(w),
                rho_centered=(
                    rho_value := rho(phi_star, axis_centered, w.text, centered=True, mu_bar=mu_bar)
                ),
                signed_load_centered=signed_load_toward_pole(
                    rho_value, _word_pole(w)),
                grandfathered=True,
                admitted=True,
            )
            for w in core_words
        ]
        expansion_loads = filter_expansion_sign(
            expansion_words, delta, phi_star, axis_centered, mu_bar
        )

        survivors = [load.text for load in core_loads] + [
            load.text for load in expansion_loads if load.admitted
        ]
        survivor_set = set(survivors)
        survivor_words.extend(w for w in loaded if w.text in survivor_set)

        admitted = sum(1 for load in expansion_loads if load.admitted)
        spec_filters.append(
            SpecificationSignFilter(
                specification=specification,
                delta=delta,
                core=core_loads,
                expansion=expansion_loads,
                survivors=survivors,
                n_core=len(core_loads),
                n_expansion_admitted=admitted,
                n_expansion_rejected=len(expansion_loads) - admitted,
            )
        )

    rebalance = run_balancing(survivor_words, seed=seed).report
    return SignFilterReport(
        specifications=spec_filters,
        delta=delta,
        gender_axis_dim=int(axis_centered.shape[0]),
        n_attributes=len(attributes),
        n_reference_items=n_reference_items,
        mu_bar_norm=float(np.linalg.norm(mu_bar)),
        seed=seed,
        arbiter_primary=str(phi_star.ref),
        rebalance=rebalance,
    )


def _word_pole(word: Word) -> GenderPole:
    """Narrow a loaded Word's gender_category to a pole; neutral words carry no pole."""
    if word.gender_category == "neutral":
        raise ValueError(
            f"{word.text!r} is neutral; the load filter handles loaded words only")
    return word.gender_category


def _mean_embedding(texts: list[str], phi_star: Arbiter) -> NDArray[np.float64]:
    """Mean of the bare-word φ* embeddings for texts (raw, unnormalized geometry)."""
    return np.mean(np.vstack([phi_star.embed(text) for text in texts]), axis=0)


def _normalize(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the unit vector; raise on a zero-norm axis (degenerate, no direction)."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError(
            "gender axis has zero norm (male and female means coincide)")
    return vec / norm
