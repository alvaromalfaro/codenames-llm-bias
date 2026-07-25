# codenames-llm-bias

A research platform that plays **Codenames Duet** LLM-vs-LLM and measures **gender bias** in both
game roles. Two model seats cooperate on a 5×5 board; one gives clues, the other guesses. Half the
boards are **probe** boards carrying a verified gender *dilemma* (the clue giver can bridge to its
target through a stereotypical or a neutral word), and half are all-neutral **control** boards that
serve as the negative control. Every clue, guess, confidence ranking and reveal is persisted to
Postgres, then scored offline against a **frozen extrinsic measurement frame** — a pinned sentence
encoder and a 768-dimensional gender axis, recorded in the database so no metric can silently redefine
its own geometry.

The board bank is built by a separate, isolated tool with its own documentation:
**[`board_generator/README.md`](board_generator/README.md)**.

---

## Contents

1. [Architecture at a glance](#architecture-at-a-glance)
2. [Repository layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Quick start with Docker](#quick-start-with-docker)
5. [The board bank](#the-board-bank)
6. [The experiment](#the-experiment)
7. [Running the batch unattended](#running-the-batch-unattended)
8. [Post-batch embedding backfill](#post-batch-embedding-backfill)
9. [Analysis scripts](#analysis-scripts)
10. [Reproducing the shipped results](#reproducing-the-shipped-results)
11. [Data model](#data-model)
12. [Interactive UI](#interactive-ui)
13. [Development](#development)
14. [Related documentation](#related-documentation)

---

## Architecture at a glance

The platform has **two entry paths over one shared core**. The interactive path serves a browser
game; the headless path plays the experiment. Both drive the same engine, the same LLM service and
the same recorder, so what the experiment measures is what the UI plays.

```
                 ┌──────────────── interactive ────────────────┐
   browser ──▶  api/routes.py ──┐                              │
                                ├─▶ game_conductor ─▶ engine ──┼─▶ recorder ─▶ writer ─▶ Postgres
   scripts/run_batch.py ──▶ batch_runner ──▶ game_runner ──────┘                          │
                                    │                                                     │
                                    └─▶ llm_service ─▶ llm/client_{local,openrouter} ──▶  models
                                                                                          │
   scripts/run_*_metrics.py ──▶ backend/app/analysis/* ◀───────────────────────────────────┘
```

| Component | File | Responsibility |
|---|---|---|
| Engine | `backend/app/core/engine.py` | Duet rules: phases, keycards, reveals, timer tokens, sudden death, win/loss. Owns the game state; seed-agnostic (an RNG is injected). |
| Clue validator | `backend/app/core/clue_validator.py` | Rejects clues that are a visible word, a morphological form of one, a compound containing one, or a repeat from the clue history (WordNet-backed). |
| Conductor | `backend/app/core/game_conductor.py` | Seat-parameterised turn orchestration (engine + service + recorder). No HTTP, no persistence — the caller injects `flush` / `on_reveal` hooks. |
| LLM service | `backend/app/core/llm_service.py` | Prompt assembly from `data/prompt_templates/`, bounded clue re-sampling, and the out-of-band **measurement** ranking elicited at the same pre-resolution state. |
| Clients | `backend/app/core/llm/client{,_local,_openrouter}.py` | Provider adapters (Ollama / OpenRouter) with a bounded same-request retry over retriable errors, structured-output enforcement and degenerate-response detection. |
| Recorder | `backend/app/db/recorder.py` | Pure in-memory accumulator for one game. No DB imports. |
| Writer | `backend/app/db/writer.py` | Atomic terminal flush of a whole game in one transaction; `delete_run` tears a run down via cascade. |
| Game runner | `backend/app/core/game_runner.py` | Headless single-game driver: deterministic identity + seed derivation, run minting, provenance, the model-digest gate. |
| Batch | `backend/app/core/batch_schedule.py`, `batch_runner.py` | The deterministic 192-game calendar and the orchestration loop with two-level fault handling. |
| Analysis | `backend/app/analysis/` | Read-only metric estimators (IAE, TAC/TAI, CIT, conc-SD, TV/PA/EP) over shared geometry and a generic cluster bootstrap. |
| UI | `backend/app/api/routes.py`, `templates/` | FastAPI + Jinja + HTMX partials for a human-vs-LLM session. |

---

## Repository layout

```
codenames-llm-bias/
├── backend/
│   ├── app/
│   │   ├── analysis/          metric estimators (read-only w.r.t. the DB)
│   │   │   ├── geometry.py        frame loading, rho = cos(phi*(w), e_gen), thresholds, terciles
│   │   │   ├── perspective.py     seat -> perspective column (the two-seat inversion trap)
│   │   │   ├── clue_metrics.py    IAE + TAC/TAI (clue-giver role)
│   │   │   ├── guesser_metrics.py CIT (guesser role)
│   │   │   ├── sd_metrics.py      conc-SD (sudden-death guesser)
│   │   │   ├── skill_metrics.py   TV / PA / EP (skill, on control boards)
│   │   │   └── inference.py       generic cluster bootstrap (games are the cluster)
│   │   ├── api/routes.py      interactive endpoints
│   │   ├── core/              engine, conductor, LLM service + clients, runner, batch, provenance
│   │   ├── db/                ORM models, session, recorder, writer, ingestion, embedding backfill
│   │   ├── models/            Pydantic schemas (game + LLM I/O) and the LLM error taxonomy
│   │   ├── templates/         Jinja templates and partials
│   │   ├── config.py          model roster + pinned local weight digests
│   │   └── main.py            FastAPI entrypoint (startup ingestion)
│   ├── migrations/            Alembic (head: 0005_run_delete_cascade)
│   ├── alembic.ini
│   └── Dockerfile
├── board_generator/           standalone board-bank tool — see its own README
├── data/
│   ├── boards/                the board bank + measurement_frame.json + balance_report.json
│   └── prompt_templates/      system / user / one-shot templates, incl. the measurement prompts
├── scripts/                   batch orchestrator CLI, analysis CLIs, embedding backfill
├── tests/unit/                platform test suite (DB-backed tests gated on DATABASE_URL)
├── docker-compose.yml         db (pgvector) + ollama (GPU) + app (FastAPI)
├── ollama_entrypoint.sh       pulls every model named in backend/app/config.py
├── exp_data.sql               pg_dump of the completed experiment (see Reproducing)
├── pyproject.toml             runtime deps + `dev` and `embeddings` extras
└── .env.example               environment template
```

---

## Prerequisites

* **Docker** with Compose v2.
* An **NVIDIA GPU** plus the NVIDIA Container Toolkit — the `ollama` service reserves one GPU device
  in `docker-compose.yml`. Without it, drop that `deploy:` block (inference then runs on CPU and a
  full batch becomes impractical) or point `OLLAMA_HOST` at an external daemon.
* **Python 3.12+** on the host for the batch and analysis CLIs. [`uv`](https://docs.astral.sh/uv/)
  is recommended; plain `pip install -e .` works too.
* Disk: the four default models are ~35 GB of weights under `docker_volumes/ollama_data/`.
* An **OpenRouter API key** only if you enable API-hosted seats (the shipped roster is all-local).

---

## Quick start with Docker

```sh
# 1. Environment. Fill in the Postgres credentials; the rest can stay as-is.
cp .env.example .env
$EDITOR .env

# 2. Bring the stack up. First run pulls the pgvector image and every model in config.llm_models,
#    so it takes a while — the ollama container logs each pull.
docker compose up -d
docker compose logs -f ollama          # watch the model pulls
docker compose ps                      # db, ollama, app

# 3. Create the schema. Migrations are NOT applied automatically.
docker compose exec app alembic -c backend/alembic.ini upgrade head

# 4. Restart the app so its startup ingestion sees the tables, then open the UI.
docker compose restart app
docker compose logs -f app             # "Measurement-frame ingestion complete", "Board ingestion complete (N new boards)"
```

The UI is then at **<http://localhost:8000>** (`/about` explains the game, `/config` starts one).

**What each service does**

| Service | Container | Port | Notes |
|---|---|---|---|
| `db` | `codenames-db` | 5432 | `ankane/pgvector`. Data in `./docker_volumes/postgres_data` (root-owned, gitignored). |
| `ollama` | `codenames-ollama` | 11434 | Runs `ollama_entrypoint.sh`, which greps `"name:tag"` strings out of the bind-mounted `backend/app/config.py` and `ollama pull`s each one. **The roster in `config.llm_models` is the pull list** — add a model there and restart the container to fetch it. |
| `app` | `codenames-backend` | 8000 | `uvicorn --reload` over the repo bind-mounted at `/workspace`. Reads `.env`. |

**Startup ingestion.** `backend/app/main.py` ingests `data/boards/measurement_frame.json` first (the
board FK target) and then every board under `data/boards/`. Both are idempotent and *defensive*: if
the database is unreachable the failure is logged as a warning and startup continues — so always
check the logs rather than assuming ingestion happened. Boards must be in the database before games
can be persisted; `writer.persist_game` refuses to write a game whose board row is absent.

> **Host vs container hostnames.** Inside Compose the database is `db:5432` and Ollama is
> `codenames-ollama:11434` — that is what `.env` holds. The batch and analysis CLIs are normally run
> **from the host**, where the same services are `localhost:5432` and `localhost:11434`. Every
> host-side snippet below overrides those two variables accordingly; forgetting to do so is the most
> common failure.

**Host-side Python environment**

```sh
uv sync                     # or: python -m venv .venv && .venv/bin/pip install -e .
uv sync --extra dev         # + pytest
uv sync --extra embeddings  # + sentence-transformers 5.5.1, only for the embedding backfill
```

---

## The board bank

`data/boards/` holds the sealed stimulus set. It currently contains **28 boards** — 14 probe
(8 `gender-career`, 6 `gender-science`) and 14 all-neutral control — plus two sidecars that the
loader deliberately skips:

* **`measurement_frame.json`** — the frozen measurement frame:
  `frame_id = 8a404797b3e656dd00683910aa829bbbc584c6b23c37e9f0de1173d11a9d0cc3`, encoder
  `sentence-transformers/all-mpnet-base-v2@e8c3b32edf5434bc2275fc9bab85f82640a19130`, mean pooling,
  normalized, with the 768-d gender axis and its construction provenance. It is content-addressed and
  immutable: ingestion raises `StaleFrameError` rather than overwriting a contradicting row.
* **`balance_report.json`** — the generator's bank-level covariate-balance report.

Each board file is a self-describing artifact (`board_id`, `type`, `category`, `specification`,
`seed`, `arbiters`, `dilemma`, `keycard_audit` and exactly 25 cards with their per-seat roles,
`weat_set` and covariates). This JSON **is** the contract between the two halves of the project:
`board_generator` writes it, `backend/app/core/loader.py` reads it, and there are no imports in
either direction.

To regenerate or extend the bank — the dilemma workflow, the manifest, determinism, the arbiter
consensus — see **[`board_generator/README.md`](board_generator/README.md)**. Nothing in this README
duplicates it.

---

## The experiment

**The schedule** (`backend/app/core/batch_schedule.py`) is deterministic and total:

* 4 models, all cross-model pairs, no self-play → **C(4,2) = 6 pairings**. A model's identity is
  `provider:model_name`; pairing keys are sorted, so reordering the input models cannot change a
  pairing's ordinal `p`.
* every pairing plays **the same 16 boards** — 4 career + 4 science + 8 control, taken in `board_id`
  ascending order — which is what makes pairings comparable;
* every board is played **twice per pairing**, mirroring which model sits in seat 0;
* `game_index = p*32 + j*2 + o` over `p ∈ 0..5`, board ordinal `j ∈ 0..15`, orientation `o ∈ {0,1}`
  → **192 games**, 50/50 probe/control.

**Determinism.** `game_id = uuid5(ns, f"{master_seed}:{game_index}")`, and every per-call seed is
derived by SHA-256 from it (`seed_game`, `seed_engine`, `seed_play(seat, turn)`,
`seed_meas(seat, turn)`). `run_id` is deliberately *not* an input to identity — it is bookkeeping. A
consequence worth internalising: re-running the same `(master_seed, game_index)` re-derives the same
`game_id` and therefore **collides on the primary key**. That is the intended "this experiment is
already recorded" signal; to legitimately re-run, delete the run first (see below).

**The digest gate.** `config.EXPECTED_LOCAL_DIGESTS` pins the exact local weights the batch is
validated against. On the batch path the served digests are resolved from the Ollama daemon once, at
run minting, and any mismatch — or an unpinned model, or an unreachable daemon — raises
`ModelDigestMismatchError` **before a single game is dispatched**. The interactive path leaves the
snapshot as a record-only witness.

**Provenance.** Each `run` row records `code_version` (git short SHA, `-dirty` suffixed),
`prompt_template_version` (a fingerprint over the loaded templates), `model_registry_snapshot` (the
served digest per seat), the `master_seed` and the `temperature`.

---

## Running the batch unattended

The batch is a long, sequential, GPU-bound job: 192 games, each many model calls, plus an extra
out-of-band measurement call per guessing turn. On four local models expect **many hours**. Run it
detached and let the preconditions protect you.

### 1. Pre-flight

```sh
cd /path/to/codenames-llm-bias

# Environment: load .env, then point at the services as seen FROM THE HOST.
set -a; source .env; set +a
export OLLAMA_HOST="http://localhost:11434"
export DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"

# Schema + boards present?  (both must be non-zero)
psql "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}" \
  -c "select count(*) as boards from board; select count(*) as frames from measurement_frame;"

# Models served and pinned?
curl -s localhost:11434/api/tags | python3 -m json.tool | head -30

# Full reproducibility check of the schedule + loop wiring: deterministic mock clients,
# no database, no digest gate, no provider calls.
python scripts/run_batch.py --master-seed 2026 --temperature 0.8 --dry-run
```

The dry run exercises exactly the code path the real batch takes, so a green dry run means the
schedule builds, the bank is sufficient and the loop is wired. It proves nothing about the models.

### 2. Launch detached

```sh
mkdir -p logs
nohup python scripts/run_batch.py --master-seed 2026 --temperature 0.8 \
  > logs/batch_2026.log 2>&1 &
echo $! > logs/batch_2026.pid
```

`setsid` or a `tmux` session work equally well and survive a closed terminal the same way:

```sh
tmux new -s batch 'set -a; source .env; set +a; \
  OLLAMA_HOST=http://localhost:11434 \
  DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}" \
  python scripts/run_batch.py --master-seed 2026 --temperature 0.8 |& tee logs/batch_2026.log'
```

**Required flags.** `--master-seed` and `--temperature` have no defaults — reproducibility is the
point, so the batch refuses to guess. `--model provider:model` may be repeated **exactly four times**
to override the roster in `config.llm_models` (e.g. `--model ollama:llama3.1:8b`); omit it to use the
configured four.

### 3. Monitor

```sh
tail -f logs/batch_2026.log                       # per-dispatch INFO lines: game_id, turn, seat
grep -c "dispatch clue" logs/batch_2026.log       # rough progress

# games actually persisted so far
psql "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}" \
  -c "select game_status, count(*) from game where run_id = '<run_id>' group by game_status;"
```

The run id is printed at mint time and again in the final report.

### 4. Failure semantics

Two levels, kept deliberately separate:

* **Level 1 — preconditions.** Checked before any game plays; each aborts the whole batch with exit
  code `2` and zero games run: a board-bank shortfall (`ScheduleError`), a missing `DATABASE_URL`,
  any of the 192 deterministic `game_id`s already present (`BatchPreconditionError`), or a digest
  mismatch (`ModelDigestMismatchError`).
* **Level 2 — per-game fault isolation.** A game that errors is recorded, classified
  (`provider` / `model` / `collision` / `other`) and the loop continues. **Five consecutive failures
  abort that pairing only** — its remaining cells are skipped and the batch moves to the next
  pairing, because a systematic failure is a finding about one pairing, not a reason to lose the
  other five.

### 5. The report

The batch ends by printing a per-pairing table (`attempted / completed / errored / skipped`, abort
reason, error-kind histogram) and then the same content as JSON. Keep the log: that JSON is the
run's receipt.

### 6. Re-running a seed

```python
# python, with DATABASE_URL exported
from backend.app.db.writer import delete_run
print(delete_run("<run_id>"))    # DeleteRunResult(found=True, games_deleted=192)
```

`delete_run` cascades through every game, seat, turn, clue, target, LLM call, proposal, item and
reveal beneath the run. Shared ingest data (`board`, `word_card`, `measurement_frame`) is not
run-owned and is never touched. It is idempotent — deleting an absent run is a no-op.

---

## Post-batch embedding backfill

Clues are not known in advance, so the extrinsic embeddings cannot be populated at startup. After a
batch completes, backfill `embedding_mpnet` with the φ\* vectors for every distinct lowercased board
word and clue word. **Every analysis script depends on this having run.**

```sh
uv sync --extra embeddings     # sentence-transformers==5.5.1, pinned to the generator's version

set -a; source .env; set +a
export DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"

# What would be embedded — loads no model, writes nothing:
python scripts/backfill_embeddings.py --frame-id 8a404797b3e656dd00683910aa829bbbc584c6b23c37e9f0de1173d11a9d0cc3 --dry-run

# The real backfill (idempotent; safe to re-run):
python scripts/backfill_embeddings.py --frame-id 8a404797b3e656dd00683910aa829bbbc584c6b23c37e9f0de1173d11a9d0cc3
```

The encoder identity (name, revision, pooling, normalize) is read from the `measurement_frame` row,
never hardcoded, and the loaded module stack is asserted against it — a swapped checkpoint fails loud
instead of writing wrong geometry. The pinned checkpoint must already be in the local HuggingFace
cache; nothing here downloads it. `--batch-size` tunes the encode/insert batch.

---

## Analysis scripts

All of them are **strictly read-only**: one session, compute, print, no writes. They share an
environment prelude:

```sh
set -a; source .env; set +a
export DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
```

| Script | Module | Measures | Extra flags |
|---|---|---|---|
| `run_clue_metrics.py` | `analysis/clue_metrics.py` | **IAE** and **TAC/TAI** point estimates | — |
| `run_clue_inference.py` | ↑ + `analysis/inference.py` | The same, with bootstrap intervals and the `gap(b1) − gap(b3)` contrast | `--replicates`, `--seed` |
| `run_guesser_metrics.py` | `analysis/guesser_metrics.py` | **CIT** point estimates + the tercile cuts | — |
| `run_cit_inference.py` | ↑ + `analysis/inference.py` | CIT with intervals and the paired `probe−control`, `b1−b3` contrasts | `--replicates`, `--seed`, `--variant {weighted,classic}` |
| `run_sd_metrics.py` | `analysis/sd_metrics.py` | **conc-SD** with intervals, power classification and diagnostics | `--replicates`, `--seed` |
| `run_skill_metrics.py` | `analysis/skill_metrics.py` | **TV / PA / EP** on control boards, with intervals | `--replicates`, `--seed` (no `--frame-id`) |
| `diagnose_skill_by_board_type.py` | ↑ (imported, not redefined) | Auxiliary: TV/PA/EP stratified by board type, with the paired `probe−control` contrast | `--replicates`, `--seed` (no `--frame-id`) |

Common flags: `--frame-id` (defaults to the live frame `8a404797…`), `--master-seed` (defaults to
`2026`, selecting the batch), `--json` for machine-readable output, `--verbose` to log progress to
stderr. Bootstrap defaults are `B = 2000` replicates, seed `2026`, percentile CI (2.5, 97.5).

**What each metric is**

* **IAE** — *implicit association effect*, clue-giver role. On a dilemma turn the giver can bridge to
  its target through a stereotypical or a neutral word; IAE is the share of **resolved** dilemmas
  taken stereotypically. Dilemmas the giver neither resolves nor separates are excluded, and the
  exclusion rate is printed next to the ratio rather than hidden inside it. Reference point: 0.5
  (indifference) — printed for scale, not as a null being tested.
* **TAC/TAI** — the rate at which a giver packs two of its own agent words into one clue, split by
  whether the pair is gender-congruent or incongruent and banded by thematic proximity. The
  reportable signal is the gap `TAC − TAI` and how it moves across bands.
* **CIT** — guesser role. A weighted Cliff-style sign delta on `[0, 1]` asking whether cards whose
  gender load is congruent with the clue's polarity are ranked above incongruent ones. **0.5 means no
  association**, not zero. It reads the out-of-band **measurement** ranking
  (`guess_proposal.kind='measurement'`), never the play proposal, so it reports belief uncontaminated
  by game strategy.
* **conc-SD** — the sudden-death analogue of CIT, where the giver is silent and the guesser works from
  the clue *history* it received. The unit is the game, clue polarity is a history mean, and thematic
  proximity is measured against the whole history. Pre-registered power rule: **PRIMARY** for a model
  only with ≥ 20 admissible sudden-death games, **EXPLORATORY** below that, applied mechanically.
* **TV / PA / EP** — skill, measured on the control boards so that competence is scored where the
  gender manipulation is absent. TV is the win rate (reported per *pairing*, the honest unit, since
  Duet is cooperative), PA the guess accuracy, EP the agents revealed per clue given (a ratio, may
  exceed 1).

**Conventions that carry the results**

* Games are the **resampling unit** — turns inside a game share a board, a keycard and a partner.
* Tercile cuts are computed once over the full data and **frozen** into the estimator closures before
  any resampling; bands cannot drift with the resample.
* Contrasts are **paired**: accumulated inside each replicate from the same draw, never differenced
  across independent intervals.
* Degenerate replicates are **dropped and counted**, never imputed; a CI resting on a minority of
  replicates is flagged `reliable: false`.
* Undefined is printed as `-`, never as `0.0` or `0.5` — "no data" and "measured, no effect" are
  different claims.
* **No p-values and no hypothesis tests.** Each cell reports an effect, a percentile interval and a
  minimum detectable effect as a sensitivity bound.
* Admissibility thresholds (`TAU_P`, `TAU_RHO` = 0.05) are pre-registered in `analysis/geometry.py`,
  not tuned.

---

## Reproducing the shipped results

`exp_data.sql` is a `pg_dump` of the completed experiment — schema **and** data, stamped
`alembic_version = 0005_run_delete_cascade`. It contains the real run (`master_seed = 2026`,
`temperature = 0.8`, `code_version = 26a115a`, four local Ollama seats: `llama3.1:8b`, `gemma4:12b`,
`qwen2.5:14b`, `mistral-small3.2:24b`), the boards and frame, and the populated `embedding_mpnet`
table. Restoring it is the fast path: every analysis script then runs against its defaults, with no
GPU and without replaying 192 LLM games.

```sh
set -a; source .env; set +a

# Restore into an EMPTY database. The dump creates its own tables and the pgvector extension;
# it contains no CREATE DATABASE, so create the target first.
createdb -h localhost -U "$POSTGRES_USER" codenames_exp
psql -h localhost -U "$POSTGRES_USER" -d codenames_exp -v ON_ERROR_STOP=1 -f exp_data.sql

export DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/codenames_exp"

python scripts/run_clue_metrics.py
python scripts/run_clue_inference.py
python scripts/run_guesser_metrics.py
python scripts/run_cit_inference.py
python scripts/run_sd_metrics.py
python scripts/run_skill_metrics.py
python scripts/diagnose_skill_by_board_type.py
```

> Restore into a **fresh** database. The dump issues bare `CREATE TABLE`s, so replaying it over a
> database that already has the schema fails partway and leaves a half-loaded mess. Do not restore
> over a database holding a run you care about.

Add `--json` to any script to capture machine-readable output for downstream aggregation.

---

## Data model

18 tables, grouped by what they are for. Full column-level documentation lives in the docstrings of
`backend/app/db/models.py`.

| Group | Tables | Purpose |
|---|---|---|
| Provenance | `run` | One experimental batch: master seed, temperature, code version, prompt-template fingerprint, served-model snapshot. |
| Measurement | `measurement_frame`, `extraction_recipe` | The frozen extrinsic frame (encoder identity + gender axis + construction) and the versioned intrinsic-extraction recipe. |
| Stimulus | `board`, `word_card` | Ingested board artifacts and their 25 cards with per-seat roles, WEAT sets and covariates. |
| Play | `game`, `game_seat`, `turn`, `clue`, `clue_target`, `reveal_event` | One row per game / seat / turn / clue / resolved intended target / engine card resolution. |
| Belief | `guess_proposal`, `guess_proposal_item` | The guesser's ordered proposals with self-reported confidence, split by `kind` into `play` and `measurement`. |
| Telemetry | `llm_call` | Per-invocation sampling telemetry for every model call. |
| Geometry | `embedding_mpnet`, `embedding_llama`, `embedding_mistral`, `word_load` | Extrinsic φ\* vectors keyed by `(frame, text)`, intrinsic vectors keyed by `(recipe, text)`, and derived per-word gender load `rho`. |

**Migrations.** Alembic is wired to `Base.metadata` and reads `DATABASE_URL` from the environment.

```sh
alembic -c backend/alembic.ini upgrade head      # apply (head: 0005_run_delete_cascade)
alembic -c backend/alembic.ini current           # inspect
alembic -c backend/alembic.ini revision --autogenerate -m "..."   # after editing db/models.py
```

Run from the repo root (`prepend_sys_path = .`), and review autogenerated revisions by hand —
pgvector columns and the cascade rules do not always round-trip.

---

## Interactive UI

`GET /` landing, `GET /about` the rules, `GET /config` the setup form (`GET /config/models` fills the
model list per provider), `GET /play` starts a game against a chosen model on a board drawn from the
selected category. During play the human acts through `POST /play/{game_id}/{clue,guess,pass}` and
the model through `POST /play/{game_id}/{llm-clue,llm-guess,llm-sd-guess}`; each returns rendered
partials that patch the board in place.

Duet phases are `giving_clue`, `guessing`, `sudden_death_human`, `sudden_death_llm`, `game_over`.
When the timer tokens run out the game enters sudden death: the human hunts their remaining agents
first, then the model hunts its own, with no clues available to either.

Two deliberate differences from the headless path: the interactive path **never enforces model
digests** (the registry snapshot stays a record-only witness), and a persistence failure at game end
is logged and swallowed so it cannot change the HTTP response. Interactive games are still written to
the same tables, under a minimal run row — filter them out by `run_id` before analysing a batch.

---

## Development

```sh
uv sync --extra dev
pytest                       # from the repo root; asyncio_mode = auto
pytest tests/unit/test_engine.py -q
```

Nine test modules are gated on `DATABASE_URL` (`pytest.mark.skipif`) and are skipped without it. When
you do enable them, point them at a **throwaway database** — they create, write and delete rows:

```sh
createdb -h localhost -U "$POSTGRES_USER" codenames_test
DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/codenames_test" \
  alembic -c backend/alembic.ini upgrade head
DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/codenames_test" \
  pytest
```

Never point them at the experiment database.

**Prompt templates** live in `data/prompt_templates/` — system, user and one-shot templates for the
clue giver and guesser, plus the sudden-death and out-of-band **measurement** variants. Editing one
changes the run's `prompt_template_version` fingerprint, which is exactly the point: a run is
comparable only to runs sharing its fingerprint.

**Model roster** lives in `backend/app/config.py`: `llm_models` (what the UI offers, what the batch
defaults to, and what `ollama_entrypoint.sh` pulls) and `EXPECTED_LOCAL_DIGESTS` (the weights the
batch is validated against). Changing a model means updating both.

**Devcontainer**: `.devcontainer/devcontainer.json` attaches VS Code to the Compose `app` service.

**`board_generator` has its own toolchain** — `uv run pytest`, `uv run ruff check .`, `uv run mypy`
from inside `board_generator/`. See its README.

---

## Related documentation

* **[`board_generator/README.md`](board_generator/README.md)** — the board-bank tool: the dilemma
  workflow, the manifest and bank determinism, the neutral-pool curation pipeline, the frozen arbiter
  consensus, and the board JSON schema that this platform consumes.
* Module docstrings are the detailed reference for everything measured here — in particular
  `backend/app/analysis/geometry.py` (the frozen measurement decisions),
  `backend/app/analysis/inference.py` (the bootstrap contract) and
  `backend/app/analysis/perspective.py` (the seat/perspective inversion trap).
