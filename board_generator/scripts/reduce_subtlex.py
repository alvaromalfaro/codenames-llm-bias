#!/usr/bin/env python3
"""Reduce the raw SUBTLEX-US workbook to the project's frequency reference CSV.

One-time data-prep step - run manually; NOT part of the generator runtime. Reads the raw SUBTLEX-US 
PoS/Zipf workbook (the 10 MB xlsx) and emits a small word,zipf,dom_pos CSV that lexicon.py consumes 
as a lookup table.

This keeps the heavy, source-specific xlsx parsing out of the runtime: lexicon.py only ever sees the 
reduced CSV. ALL rows are kept (it is a general frequency reference for the whole project - the WEAT 
cores, the neutral pool, and later expansions all look up against it); only columns are reduced.

Decisions:
  * frequency = SUBTLEX-US Zipf-value (the log-normalised scale, suited to a PSM covariate).
  * words are case-folded to lowercase (lexicon.py joins on lowercase).
  * case-fold collisions (e.g. US vs us) are resolved by keeping the entry with the MAX Zipf (the 
  more frequent surface form dominates); the count is reported.
  * dom_pos is the SUBTLEX dominant part of speech (e.g. Noun, Verb, Name for proper nouns); kept 
  verbatim for lexicon.py's playability check.

Usage:
    uv run python scripts/reduce_subtlex.py path/to/SUBTLEX-US-raw.xlsx -o resources/frequencies/subtlex_us.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Source column name -> output column name. Selection is by NAME, so the script is robust to extra
# columns and to column reordering in the source workbook.
COLUMN_MAP = {"Word": "word", "Zipf-value": "zipf",
              "Dom_PoS_SUBTLEX": "dom_pos"}


def reduce_subtlex(src: Path, dst: Path) -> None:
    try:
        df = pd.read_excel(src, usecols=list(COLUMN_MAP), engine="openpyxl")
    except ValueError as exc:
        # Most often: a header name does not match (e.g. 'Zipf' vs 'Zipf-value').
        available = pd.read_excel(
            src, engine="openpyxl", nrows=0).columns.tolist()
        sys.exit(
            f"could not select the expected columns from {src.name}: {exc}\n"
            f"expected {list(COLUMN_MAP)}, found {available}"
        )

    df = df.rename(columns=COLUMN_MAP)

    df["word"] = df["word"].astype("string").str.strip().str.lower()
    df["dom_pos"] = df["dom_pos"].astype("string").str.strip()
    df["zipf"] = pd.to_numeric(df["zipf"], errors="coerce")

    # Drop unusable rows: empty/NA word or missing frequency.
    df = df[(df["word"].notna()) & (df["word"] != "") & (df["zipf"].notna())]

    # Resolve case-fold collisions: keep the max-Zipf entry per word.
    n_before = len(df)
    df = df.sort_values("zipf", ascending=False).drop_duplicates(
        "word", keep="first")
    collisions = n_before - len(df)

    # deterministic output (stable diffs, reproducibility)
    df = df.sort_values("word")

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, columns=[
              "word", "zipf", "dom_pos"], float_format="%.4f")

    print(
        f"wrote {len(df)} rows to {dst} "
        f"({collisions} case-fold collision(s) resolved by max Zipf)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce the raw SUBTLEX-US xlsx to a word,zipf,dom_pos reference CSV."
    )
    parser.add_argument("src", type=Path, help="raw SUBTLEX-US .xlsx")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("resources/frequencies/subtlex_us.csv"),
        help="output CSV path (default: resources/frequencies/subtlex_us.csv)",
    )
    args = parser.parse_args()
    if not args.src.exists():
        sys.exit(f"source not found: {args.src}")
    reduce_subtlex(args.src, args.out)


if __name__ == "__main__":
    main()
