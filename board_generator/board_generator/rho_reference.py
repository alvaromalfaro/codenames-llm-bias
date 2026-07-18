"""Build the frozen per-word ρ reference fixture.

For every word card on a reference board, three signed gender scores are computed through the RAW φ*
(never ``_CenteringEncoder`` / ``axis_diagnostics``; that path carries ``ref=phi_star.ref`` over a
modified geometry and would freeze a mislabelled identity into permanent test data):

  * ``rho_raw``  = cos(φ*(w), e_gen)                                   [PRIMARY scale]
  * ``rho_cent`` = cos(φ*(w) - μ̄, e_gen) via ``load_filter.rho(centered=True, mu_bar=…)`` [robust]
  * ``rho_weat`` = s(w, A, B) = mean_{a∈A} cos(φ*(w), φ*(a)) - mean_{b∈B} cos(φ*(w), φ*(b))  [WEAT]

s(w, A, B) is the WEAT/SEAT differential association (Caliskan et al. 2017), a RAW difference of
mean cosines (NOT divided by any pooled std: a per-word score has no X/Y target sets, so the
canonical effect-size normalization does not apply; standardization happens downstream over the
full distribution). Male is positive on all three scales.

Crucially, rho_weat's lexicon A/B is the **WEAT-6 proper names only** (``WEAT_ROBUSTNESS_SET``),
which is DISJOINT from the 6U7U8 lexicon ``build_gender_axis`` pools into e_gen. Using the axis's
own lexicon would make rho_weat = const·rho_raw (unit-norm φ* -> perfect collinearity, corr 1.0),
giving no independent robustness signal; the disjoint names lexicon decouples the two scales (the
measured correlation is recorded in the fixture as ``rho_raw_weat_correlation``). This is a
robustness scale in the fixture only, it does not change e_gen, μ̄, or the frame_id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from board_generator.arbiter import Arbiter
from board_generator.load_filter import AttributeWord, rho

# The robustness lexicon for rho_weat: WEAT-6 proper names only (JOHN/AMY-style), which are
# disjoint from the axis lexicon (weat-6 shares no word with weat-7/8), so rho_weat is a
# non-collinear robustness scale, not a constant multiple of rho_raw. Does NOT change e_gen or the
# frame.
WEAT_ROBUSTNESS_SET = "weat-6"

RHO_WEAT_DEFINITION = (
    "rho_weat(w) = s(w, A_names, B_names) = mean_{a in A} cos(phi*(w),phi*(a)) "
    "- mean_{b in B} cos(phi*(w),phi*(b)); A/B = weat-6 male/female PROPER NAMES (JOHN/AMY-style), "
    "DISJOINT from the 6U7U8 axis lexicon (weat-6 shares no word with weat-7/8), so rho_weat is an "
    "independent robustness scale, not const*rho_raw; raw differential (no std normalization); "
    "male-positive. Measured decoupling is in rho_raw_weat_correlation."
)
COMPARISON_NOTE = (
    "compare rho_* with tolerance (~1e-6), NOT bit-exact: the platform re-embeds and "
    "BLAS/dtype/summation order are not bit-reproducible across the generator<->platform boundary."
)

# tests/fixtures/ under the board_generator project root (this module is the inner package).
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def attribute_pole_terms(attributes: list[AttributeWord]) -> tuple[list[str], list[str]]:
    """Sorted unique male/female words from the given attributes (dedup by word, like the axis).

    Generic partition helper: ``build_rho_reference`` passes it the WEAT-6-only robustness subset;
    it is otherwise the same (word, pole) dedup + sort ``build_gender_axis`` applies.
    """
    male = sorted({a.word for a in attributes if a.gender_pole == "male"})
    female = sorted({a.word for a in attributes if a.gender_pole == "female"})
    return male, female


def weat_association(
    word_vec: NDArray[np.float64],
    male_vecs: NDArray[np.float64],
    female_vecs: NDArray[np.float64],
    phi_star: Arbiter,
) -> float:
    """s(w, A, B): mean cos(w, male) - mean cos(w, female). Male-positive; raw (no std norm).

    ``phi_star.cos`` renormalizes internally, which is correct: the attribute vectors are raw model
    vectors too. Inputs are raw φ* embeddings (``Arbiter.embed``), never centered.
    """
    male_mean = float(np.mean([phi_star.cos(word_vec, a) for a in male_vecs]))
    female_mean = float(
        np.mean([phi_star.cos(word_vec, b) for b in female_vecs]))
    return male_mean - female_mean


def build_rho_reference(
    *,
    board_id: str,
    card_texts: list[str],
    attributes: list[AttributeWord],
    phi_star: Arbiter,
    gender_axis: NDArray[np.float64],
    mu_bar: NDArray[np.float64],
    encoder: dict[str, Any],
    frame_id: str,
) -> dict[str, Any]:
    """Compute the fixture: per-card ``rho_raw`` / ``rho_cent`` / ``rho_weat`` through the raw φ*.

    ``gender_axis`` and ``mu_bar`` are the sidecar's own vectors (caller-verified): rho_cent
    uses the frame's single μ̄, and rho_raw/rho_cent go through ``load_filter.rho`` exactly. rho_weat
    uses the disjoint WEAT-6 names lexicon (not the axis lexicon), so it is an independent scale.
    """
    lexicon = [a for a in attributes if a.weat_set == WEAT_ROBUSTNESS_SET]
    male_terms, female_terms = attribute_pole_terms(lexicon)
    if not male_terms or not female_terms:
        raise ValueError(
            f"rho_weat robustness lexicon {WEAT_ROBUSTNESS_SET!r} needs male and female terms"
        )
    male_vecs = np.vstack([phi_star.embed(term) for term in male_terms])
    female_vecs = np.vstack([phi_star.embed(term) for term in female_terms])

    words: list[dict[str, Any]] = []
    for text in card_texts:
        word_vec = phi_star.embed(text)
        words.append(
            {
                "text": text,
                "rho_raw": rho(phi_star, gender_axis, text),
                "rho_cent": rho(phi_star, gender_axis, text, centered=True, mu_bar=mu_bar),
                "rho_weat": weat_association(word_vec, male_vecs, female_vecs, phi_star),
            }
        )

    return {
        "frame_id": frame_id,
        "board_id": board_id,
        "encoder": {
            "name": encoder["name"],
            "revision": encoder["revision"],
            "pooling": encoder["pooling"],
            "normalize": encoder["normalize"],
        },
        "rho_weat_definition": RHO_WEAT_DEFINITION,
        "rho_weat_lexicon": {
            "weat_set": WEAT_ROBUSTNESS_SET,
            "male_terms": male_terms,
            "female_terms": female_terms,
            "counts_by_pole": {"male": len(male_terms), "female": len(female_terms)},
        },
        "rho_raw_weat_correlation": _pearson(
            [w["rho_raw"] for w in words], [w["rho_weat"] for w in words]
        ),
        "comparison": COMPARISON_NOTE,
        "words": words,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length series; ``None`` if undefined (<2 pts / no variance).

    Reports how far rho_weat has decoupled from rho_raw. Not part of any hash.
    """
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def read_board_card_texts(board_path: Path) -> tuple[str, list[str]]:
    """Read a board JSON → (board_id, list of every card's ``text``), preserving card order."""
    board = json.loads(board_path.read_text(encoding="utf-8"))
    return board["board_id"], [card["text"] for card in board["cards"]]


def fixture_path(frame_id: str, out_dir: Path = FIXTURES_DIR) -> Path:
    """Fixture location keyed by frame_id's first 8 hex: ``rho_reference_<frame_id8>.json``."""
    return out_dir / f"rho_reference_{frame_id[:8]}.json"


def write_fixture(fixture: dict[str, Any], out_dir: Path = FIXTURES_DIR) -> Path:
    """Write the fixture to ``rho_reference_<frame_id8>.json`` (indent=2, LF-terminated)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = fixture_path(fixture["frame_id"], out_dir)
    payload = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path
