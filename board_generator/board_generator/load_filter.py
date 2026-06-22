"""Gender-load filter.

Finalizes the loaded word pools with a gender-load filter built on the signed load
ρ_w = cos(φ*(w), e_gen). This reuses the SAME primary arbiter φ* and gender axis the downstream
metrics use, here as a measurement instrument - not the dilemma consensus gate. ρ_w is a continuous
measure, so it uses a single fixed encoder (φ*), never a consensus average (geometries from
different encoders are not commensurable).

Grandfathering: the WEAT core (weat_set != ()) is canonical and citable, so it is never filtered;
it is exactly what calibrates the per-specification threshold τ_load. Only expansion words
(weat_set == ()) are filtered. An expansion word enters its pole only if its signed load toward
that pole clears τ_load - this rejects topically-correct but gender-weak words.

Determinism: the axis is male-minus-female, so male is positive by construction and there is no PCA
sign ambiguity. Given a fixed φ*, the axis, τ_load and the partition are deterministic.

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
    """Non-destructive diagnostic. Pure data - JSON-serializable (allow_nan=False holds).
    """

    specifications: list[SpecificationLoadFilter]
    gender_axis_dim: int
    n_attributes: int
    seed: int
    quantile: float
    arbiter_primary: str  # str(φ*.ref) - the single measurement encoder
    rebalance: BalanceReport


def read_attribute_words(path: Path) -> list[AttributeWord]:
    """Read the gender-attribute CSV (word,gender_pole,source,weat_set) into AttributeWords.

    Words are lowercased (φ*.embed lowercases too). Blank words are skipped; an unknown gender_pole
    is a hard error - a typo must never silently corrupt the axis.
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


def build_gender_axis(attributes: list[AttributeWord], phi_star: Arbiter) -> NDArray[np.float64]:
    """Build the gender axis e_gen by mean-difference.

        e_gen = normalize(mean_{w in male} φ*(w) - mean_{w in female} φ*(w))

    Uses ALL attributes (names + pronouns + kinship). Deduplicates to unique (lowercased word, pole)
    pairs first, so repeated attributes (e.g. BROTHER/SON/HE shared by weat-7 & weat-8) are not
    double-weighted. Raises if a word appears under conflicting poles. Male is positive by
    construction (axis is male - female).
    """
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

    diff = _mean_embedding(male, phi_star) - _mean_embedding(female, phi_star)
    return _normalize(diff)


def rho(phi_star: Arbiter, e_gen: NDArray[np.float64], text: str) -> float:
    """Signed gender load: ρ_w = cos(φ*(text), e_gen). Male is positive by construction."""
    return phi_star.cos(phi_star.embed(text), e_gen)


def signed_load_toward_pole(rho_value: float, pole: GenderPole) -> float:
    """Flip the male-positive load to point toward the word's own pole (female words flip sign)."""
    return rho_value if pole == "male" else -rho_value


def calibrate_tau_load(
    core_words: list[Word], phi_star: Arbiter, e_gen: NDArray[np.float64], quantile: float
) -> float:
    """Calibrate τ_load(s) from the per-spec WEAT core, anchored to the grandfathered core.

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
    """Apply τ_load to expansion words.

    An expansion word of pole P is admitted iff signed-load-toward-P(w) >= τ (non-strict: an exact
    tie passes), else REJECTED. This is what rejects topically-correct but gender-weak bridges.
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


def build_load_filter_report(
    words: list[Word],
    attributes: list[AttributeWord],
    phi_star: Arbiter,
    *,
    seed: int,
    quantile: float = DEFAULT_QUANTILE,
) -> LoadFilterReport:
    """Assemble the non-destructive report. Pure: φ* is injected.

    Builds e_gen once, then per specification splits the loaded words into grandfathered core
    (weat_set != ()) and filtered expansion (weat_set == ()), calibrates τ_load(s) on the core, and
    filters the expansion. The survivor set is all core + admitted expansion. Finally re-runs
    balancing.run_balancing over the survivors of both specifications and embeds its report.
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
