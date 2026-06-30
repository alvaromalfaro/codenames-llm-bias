"""Local φ* smoke - exercises the dilemma_flow core end-to-end.

Builds ONE dilemma through the real DilemmaSession with the frozen consensus trio: loads pools,
ranks neutral and stereotypical bridges via the session, runs the Eq. 4.1 consensus gate, assembles
a DilemmaRecord, and round-trips it through reverify to confirm consensus_ok reproduces under real
geometry. Non-interactive: it picks the top-ranked candidate of each list as a stand-in for the
manual choice, purely to drive the seam. Read-only by default (does NOT write the artifact unless
--write is passed).

Usage (from board_generator/):
    uv run python scripts/smoke_dilemma_phi_star.py --target nurse --spec gender-career
    uv run python scripts/smoke_dilemma_phi_star.py --target nurse --spec gender-career --write
    uv run python scripts/smoke_dilemma_phi_star.py --target nurse --spec gender-career --k 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from board_generator import dilemma_flow
from board_generator.arbiter import DEFAULT_CONSENSUS, Arbiter, load_consensus
from board_generator.lexicon import load_words

DEFAULT_WORDS_DIR = Path(__file__).resolve(
).parent.parent / "resources" / "words"
DEFAULT_SUBTLEX = (
    Path(__file__).resolve().parent.parent /
    "resources" / "frequencies" / "subtlex_us.csv"
)


def _primary(consensus: list[Arbiter]) -> Arbiter:
    for a in consensus:
        if a.ref == DEFAULT_CONSENSUS.primary:
            return a
    raise ValueError("primary φ* not found among loaded consensus arbiters")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    help="loaded word to use as w_target")
    ap.add_argument("--spec", required=True,
                    help="gender-career | gender-science")
    ap.add_argument("--k", type=int, default=dilemma_flow.DEFAULT_K)
    ap.add_argument("--write", action="store_true",
                    help="actually write the artifact (default: read-only smoke)")
    args = ap.parse_args()

    print(">> loading pools (lexicon)...")
    res = load_words(DEFAULT_WORDS_DIR, DEFAULT_SUBTLEX)
    loaded = res.words
    by_text = {w.text: w for w in loaded}

    target = by_text.get(args.target.lower())
    if target is None:
        print(
            f"!! target {args.target!r} not in loaded words", file=sys.stderr)
        return 1
    if target.specification != args.spec or target.gender_category not in {"male", "female"}:
        print(f"!! target {target.text!r} is gender={target.gender_category}, "
              f"spec={target.specification}; need a male/female word of spec {args.spec!r}",
              file=sys.stderr)
        return 1
    print(
        f"   target={target.text!r} ({target.gender_category}, spec={target.specification})")

    print(">> loading the frozen consensus trio (Hugging Face - may download on first run)...")
    consensus = load_consensus(DEFAULT_CONSENSUS)
    phi_star = _primary(consensus)
    print(f"   φ* = {phi_star.ref}")
    print(f"   consensus = {[str(a.ref) for a in consensus]}")

    session = dilemma_flow.DilemmaSession(
        phi_star=phi_star,
        consensus=consensus,
        specification=args.spec,
    )

    print(f">> session.rank_neutral (top-{args.k})...")
    neutral_ranked = session.rank_neutral(target, loaded, k=args.k)
    for w, c in neutral_ranked:
        print(f"     {c:+.4f}  {w.text}")

    print(f">> session.rank_stereo (top-{args.k}, gender-congruent)...")
    stereo_ranked = session.rank_stereo(target, loaded, k=args.k)
    for w, c in stereo_ranked:
        print(f"     {c:+.4f}  {w.text}  [{w.gender_category}]")

    if not neutral_ranked or not stereo_ranked:
        print("!! empty ranking - cannot form a triple", file=sys.stderr)
        return 1

    # Top-ranked stand-ins for the manual selections (drives the gate; NOT the real flow).
    neutral = neutral_ranked[0][0]
    stereo = stereo_ranked[0][0]
    print(
        f">> session.attempt  (neutral={neutral.text!r}, stereo={stereo.text!r})...")
    dilemma = session.attempt(target, neutral, stereo)
    for s in dilemma.arbiter_scores:
        mark = "OK  " if s.satisfies_eq_4_1 else "FAIL"
        print(f"     [{mark}] {s.arbiter}")
        print(f"            cos(t,neutral)={s.cos_target_neutral:+.4f}  "
              f"cos(t,stereo)={s.cos_target_stereo:+.4f}")
    print(
        f">> consensus_ok = {dilemma.consensus_ok}   rejected so far = {len(session.rejected)}")

    if not dilemma.consensus_ok:
        print("   (top-neutral vs top-stereo did NOT pass the gate - interesting case; in the real "
              "flow you would re-select. Smoke stops here.)")
        return 0

    record = session.build_record(dilemma)
    print(f">> built DilemmaRecord  id={dilemma_flow.record_id(args.spec, target.text)!r}  "
          f"attempts_count={record.attempts_count}  rejected={len(record.rejected_attempts)}")

    # The reproducibility check: reverify from the record's stored selections under real geometry.
    print(">> reverify(record, consensus) - must reproduce consensus_ok...")
    rev = dilemma_flow.reverify(record, consensus)
    ok = rev.consensus_ok == record.accepted.consensus_ok
    same_cos = all(
        a.cos_target_neutral == b.cos_target_neutral
        and a.cos_target_stereo == b.cos_target_stereo
        for a, b in zip(rev.arbiter_scores, record.accepted.arbiter_scores)
    )
    print(
        f"   reverify consensus_ok matches: {ok}   per-arbiter cosines identical: {same_cos}")
    if not (ok and same_cos):
        print("!! reverify diverged - reproducibility broken", file=sys.stderr)
        return 1

    if args.write:
        path = dilemma_flow.write_record(record)
        print(f">> wrote artifact: {path}")
    else:
        print(">> read-only smoke (pass --write to serialize the artifact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
