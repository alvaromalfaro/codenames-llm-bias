# board_generator

A standalone tool that builds a fixed **bank of Codenames Duet boards** instrumented to measure
**gender bias** in LLMs. It loads curated word lists, balances their confound covariates, builds
verified gender *dilemmas*, lays out 5×5 double-sided key cards, and serializes each board to JSON.

**Isolation is the design.** The tool writes only under `board_generator/` (its own
`resources/`) and — as its single point of contact with the game platform — emits board files into
the repo-root `data/boards/` directory. The platform *reads* those files; there are no cross-imports
in either direction. Nothing else is shared.

## What it produces

A bank of **`2·P` boards** for `P` pre-built dilemmas: `P` **probe** boards + `P` **control**
boards, 50/50 by construction, plus one bank-level `balance_report.json`.

- **Probe** — carries a 3-word gender *dilemma* (target / stereotypical bridge / neutral bridge).
  Composition (`n_pairs = 6`): **3 dilemma + 12 loaded (6 male + 6 female PSM pairs) + 10 neutral =
  25** cards.
- **Control** — an all-neutral baseline with no dilemma: **25 distinct neutral** cards.

Each probe's dilemma is verified once (interactively, against the arbiter consensus) and then
consumed verbatim; the bank build never re-runs the arbiters.

## The two modes

The CLI (`board-generator`) has two subcommands of opposite natures:

| | `dilemma` | `bank` |
|---|---|---|
| Nature | **Interactive**, one dilemma at a time | **Offline batch**, whole bank |
| Needs Hugging Face / φ* | **Yes** (loads encoders) | **No** |
| Determinism | semi-automatic (manual picks) | byte-reproducible |
| Output | `resources/dilemmas/dilemma_<spec>_<target>.json` | boards + `balance_report.json` in `data/boards/` |

- **`dilemma`** ranks candidate bridges by cosine to the primary arbiter **φ\***, takes the three
  manual expert selections, and verifies **Eq. 4.1** (`cos(target, neutral) ≥ cos(target, stereo)`
  for *all* consensus arbiters). Accepted dilemmas are written as intermediate artifacts.
- **`bank`** consumes those artifacts plus a manifest. It is fully offline and never touches φ*: a
  dilemma's `consensus_ok` is trusted from its artifact, never recomputed. The same manifest over
  the same word pools yields the same bank, byte for byte.

## Setup

Run everything from inside `board_generator/` (requires Python ≥ 3.12; [uv](https://docs.astral.sh/uv/) recommended):

```sh
uv sync                                    # install runtime + dev deps (creates the venv)
uv run python -m nltk.downloader wordnet   # one-time: WordNet corpus (polysemy covariate)

uv run pytest                              # offline test suite (skips integration by default)
uv run pytest -m integration               # integration tests (download a real HF model)

uv run ruff check .                        # lint
uv run mypy                                # type-check
```

`uv run ruff check .` and `uv run mypy` are the gates. (`ruff format --check` is not run as a gate.)

## End-to-end workflow

```sh
cd board_generator

# 1. Build one verified dilemma per probe board (interactive; needs HF / φ*).
#    Repeat per target; each writes resources/dilemmas/dilemma_<spec>_<target>.json
uv run board-generator dilemma --spec gender-career
uv run board-generator dilemma --spec gender-science
#    Tune the candidate list / search budget if needed:
#    uv run board-generator dilemma --spec gender-career --k 12 --attempt-cap 50

# 2. Write a manifest listing the dilemma artifacts in order (see below), e.g. manifest.json.

# 3. Dry-run: build + validate the whole bank, write nothing.
uv run board-generator bank --manifest manifest.json --dry-run

# 4. Build the bank for real (boards + balance_report.json → ../data/boards/).
uv run board-generator bank --manifest manifest.json

# 5. Confirm byte-reproducibility: build twice and diff.
uv run board-generator bank --manifest manifest.json --out-dir /tmp/bank_a
uv run board-generator bank --manifest manifest.json --out-dir /tmp/bank_b
diff -r /tmp/bank_a /tmp/bank_b      # expect no differences

# 6. Confirm the JSON contract against the platform reader (loads data/boards/*.json).
#    From the repo root:
#    uv run pytest tests/unit/test_loader.py
```

`board-generator bank --help` / `board-generator dilemma --help` print the full flag list. `bank`
flags: `--manifest` (required), `--words-dir` (default `resources/words`), `--subtlex-path` (default
`resources/frequencies/subtlex_us.csv`), `--dilemmas-dir` (default `resources/dilemmas`), `--out-dir`
(default `../data/boards`), `--dry-run`. `dilemma` flags: `--spec {gender-career,gender-science}`
(required), `--words-dir`, `--subtlex-path`, `--out-dir` (default `resources/dilemmas`), `--k`
(default 8), `--attempt-cap` (default: unlimited).

> `resources/dilemmas/` does not exist until the first `dilemma` run creates it.

## Manifest

The manifest is the **single source of truth** for the bank — same manifest + same word pools ⇒
byte-identical bank. Its schema (validated by `bank.load_manifest`):

```json
{
  "master_seed": 20260629,
  "dilemmas": [
    "dilemma_gender-career_executive.json",
    "dilemma_gender-science_physics.json"
  ]
}
```

- `master_seed` *(int)* — the root of all bank determinism.
- `dilemmas` *(list of filenames)* — resolved against `--dilemmas-dir`. **Order matters**: within
  each specification, the artifacts assign each probe board its index (`probe-<spec>-000`,
  `probe-<spec>-001`, …). One control board (`control-000`, …) is emitted per dilemma.

Every per-board random draw derives from a **per-board seed** =
`derive_board_seed(master_seed, board_id)` — the first 8 bytes of `sha256("<master_seed>:<board_id>")`
as a big-endian integer. Because the seed depends only on the board *id*, reordering the manifest
never changes a given board's seed. The bank-level balancing runs once, seeded by `master_seed`.

## Neutral pool

The neutral fill is built **upstream**, by hand, and is an **input** to composition — the bank loop
never produces it. The pipeline: the Duet deck (`resources/neutral_pool/duet.txt`) → a read-only,
recall-oriented enumerator (`neutral.py`, surfaced via `scripts/build_neutral_candidates.py`) that
*flags* denotational-gender candidates but never decides inclusion → a manual **denotational
stoplist** (`resources/curation/gender_denotational_stoplist.csv`) → a manual connotational review →
the curated `resources/words/neutral.csv` that the lexicon loads. Provenance for the candidate set is
recorded in `resources/neutral_pool/neutral_candidates.provenance.json`.

## Output JSON schema

`board.write_board` writes one file per board to `--out-dir` named
`f"{bias_category}_{board_id}.json"` — e.g. `gender_probe-gender-career-000.json`,
`neutral_control-000.json`. The file *is* the platform contract and is validated defensively before
being written.

Board-level keys (in order):

```
board_id, type, category, specification, seed,
grid { rows, cols },
arbiters { consensus, primary },          # each entry "model@rev"
dilemma | null,                           # probe only; null for control
keycard_audit { per_perspective, overlap_ok, role_gender_independent },
cards [ ... ]                             # exactly 25
```

Card-level keys (in order):

```
id, text, human_perspective_role, llm_perspective_role,
category, source, weat_set [ ... ],
covariates { subtlex_freq, length, wordnet_polysemy }
```

**Contract translations** applied by `to_json_dict` (internal representation → platform contract):

- internal role `bystander` → **`civilian`**, on the two card role fields only
  (`keycard_audit.per_perspective` keeps its internal `bystander` key verbatim);
- card `text` is **UPPERCASED** (single token);
- an out-of-vocabulary neutral's `subtlex_freq` stays JSON **`null`** (imputation is balancing's job,
  not serialization's);
- board-level **`category` = bias axis** (`gender` for probe, `neutral` for control) vs card-level
  **`category` = pole** (`male` | `female` | `neutral`).

This matches the platform reader (`backend/app/core/loader.py` globs `data/boards/*.json`;
`backend/app/models/game_schemas.py` requires `board_id`, `category`, and exactly 25 cards with
single-token `text` and roles in `{agent, assassin, civilian}`).

## Layout

```
board_generator/
  pyproject.toml              tool packaging; console script board-generator = cli:main
  board_generator/            import package (flat layout, no src/)
    lexicon.py                word CSV loading, dedup, covariate annotation, playability validation
    balancing.py              propensity-score matching + per-covariate equivalence (SMD governing)
    roles.py                  double-sided key-card assignment + legality / independence audit
    dilemma.py                candidate ranking + Eq. 4.1 verification
    dilemma_flow.py           φ*-agnostic dilemma session + DilemmaRecord artifact I/O
    composition.py            select the 25 words (probe / control)
    arbiter.py                external prefix-free encoders; frozen consensus + primary φ*
    board.py                  schema dataclasses, board assembly, JSON serialization
    bank.py                   master bank loop (manifest → boards + balance report)
    cli.py                    argparse wiring + interactive prompts (exposes main())
    neutral.py                read-only denotational-gender candidate enumerator (curation aid)
    load_filter.py            gender-axis load / sign filtering diagnostics
    axis_diagnostics.py       axis diagnostics support
  resources/                  tool INPUTS
    words/                    gender_career.csv, gender_science.csv, neutral.csv
    attribute_words/          gender_attributes.csv (WEAT attribute words)
    curation/                 gender_denotational_stoplist.csv (manual stoplist)
    frequencies/              subtlex_us.csv (+ SUBTLEX-US-raw.xlsx) — lexical frequency
    neutral_pool/             duet.txt, neutral_candidates.csv (+ .provenance.json)
    dilemmas/                 verified dilemma artifacts (created on first `dilemma` run)
  scripts/                    offline reports: balance, load filter, neutral candidates/audit, ...
  tests/                      one test per invariant (offline; `-m integration` for HF-backed)

../data/boards/               emitted board files — the only coupling with the platform
```

## Key design rules

- **Isolation.** Code lives under `board_generator/`; the only output coupling is board files in
  `data/boards/`. No imports cross the boundary either way.
- **Gender only, binary axis.** A documented simplification: `gender_category ∈ {male, female,
  neutral}`.
- **Manual steps stay manual.** The three dilemma selections (target / neutral bridge /
  stereotypical bridge) and the neutral-pool stoplist + connotational review are expert decisions;
  the tool ranks and verifies, the human decides.
- **Frozen external arbiter consensus.** Three prefix-free symmetric encoders pinned by HF revision
  — `all-mpnet-base-v2` (the primary **φ\***), `gte-large`, and `sentence-t5-large` — recorded in
  each board as `model@rev`. No model under evaluation may serve as an arbiter. Arbiters are used at
  build time only; they never admit or reject words.
- **Determinism.** Every random draw (key card, word placement, pair/neutral selection) derives from
  the per-board seed, itself derived from `master_seed`. The bank is reproducible byte for byte.
