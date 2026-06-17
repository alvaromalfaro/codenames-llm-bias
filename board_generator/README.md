# board_generator

Standalone tool that generates a **fixed bank of `T = 30` Codenames Duet boards**,
instrumented to measure **gender bias** in LLMs.

It is **isolated** from the game platform: the platform only *reads* the emitted board
files from the shared repo-root `data/boards/` directory. There are no cross-imports in
either direction.

> **Status: scaffold.** The package is laid out as typed stubs (`NotImplementedError`).
> Only the pure validators (key-card legality, grid shape, consensus-spec checks) and the structural 
> invariant tests are implemented so far.

## Commands

Run from inside `board_generator/`:

```sh
uv sync                                   # install deps (creates the venv)
uv run python -m nltk.downloader wordnet  # one-time: WordNet corpus (polysemy covariate)
uv run board-generator                    # interactive semi-automatic flow (stub)
uv run pytest                             # tests
uv run ruff check . && uv run ruff format .
uv run mypy
```

## Layout

```
board_generator/        import package (flat layout, no src/)
  lexicon.py/           word loading + covariate annotation
  balancing.py          PSM + equivalence checks
  roles.py              double-sided key-card assignment + validation
  dilemma.py            candidate ranking + Eq. 4.1
  arbiter.py            external prefix-free encoders, consensus + φ*
  board.py              schema, assembly, serialization
  cli.py                interactive flow (exposes main())
resources/              tool INPUTS (WEAT, Eurostat, She Figures, SUBTLEX-US)
tests/                  one test per invariant
```

Boards are written to the repo-root `data/boards/` (default `../data/boards/`,
configurable) - the single coupling point with the platform.
