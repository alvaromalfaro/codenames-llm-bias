"""Play N games between two models and write a full log of each one to a text file.

A diagnostic tool, not the batch: no database, no schedule, no digest gate unless asked. Each game is
played by ``run_single_game``, the same driver the batch uses, so what is logged is what the batch
would play. The log is meant to be read after the run to find anything that went wrong.

What the log holds (one file per run, ``logs/games_<timestamp>.log``):

  * the run: command line, git commit and uncommitted changes, Python, the Ollama server version and
    the digest of each model (checked against config.EXPECTED_LOCAL_DIGESTS), the seeds, and the
    full text of every prompt template, each with the id the call log uses for it;
  * each game: the board with both sides of the key card, the seats, and then every event in order:
    each model call (the rendered prompt and the raw response, with latency and tokens), each clue
    and whether it is valid, each guess with what the card is on both sides, skipped proposals,
    retries, stops, turn changes, sudden death and the end of the game;
  * after every turn, the state (tokens, pending words, covered words, time tokens) and a check of
    the game's invariants: the 9 tokens add up, pending_words matches the board, the words visible
    to the clue validator are the uncovered ones, every time token sits on a word that was beige for
    the clue giver, the phase fits the reserve and the pending words. A broken invariant is logged
    as an ERROR;
  * at the end of each game, the final board and the clues given; at the end of the run, a summary
    with every warning and error, game by game.

The console only shows the progress and the summary. The exit code is 1 if a game ended in error or
broke an invariant, 0 otherwise.

Environment: run from the REPO ROOT with an Ollama server up (OLLAMA_HOST, default
http://localhost:11434), or OPENROUTER_API_KEY for OpenRouter models.

Examples:
    # 4 games, llama3.1 against qwen2.5, swapping seats every other game:
    python scripts/play_games.py --model-a ollama:llama3.1:8b --model-b ollama:qwen2.5:14b --games 4

    # one model against itself, on the control boards only:
    python scripts/play_games.py --model-a ollama:llama3.1:8b --games 2 --board-type control
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

from backend.app import config
from backend.app.core.engine import CodenamesDuetEngine
from backend.app.core.game_runner import GameRunResult, SeatSpec, run_single_game
from backend.app.core.llm_service import LLMService, text_digest
from backend.app.core.loader import BoardLoader
from backend.app.db.recorder import GameRecorder
from backend.app.models.game_schemas import Board, CardRole, GamePhase, WordCard

logger = logging.getLogger("play_games")

_BOARD_DATA_PATH = "data/boards"
_KNOWN_PROVIDERS = ("ollama", "openrouter")
_RESULTS = {"victory", "loss_assassin", "victory_sd", "loss_civilian_sd", "loss_assassin_sd",
            "loss_stopped_sd"}
# Libraries whose own DEBUG/INFO chatter would bury the game in the log.
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "asyncio", "openai", "ollama", "sqlalchemy")


# logging setup

class _GameContext(logging.Filter):
    """Stamps every record with the game being played ("run" outside a game), so each line of the
    log says which game it belongs to, whichever module wrote it."""

    label = "run"

    def filter(self, record: logging.LogRecord) -> bool:
        record.game = self.label
        return True


class _IssueCollector(logging.Handler):
    """Keeps every WARNING and ERROR, by game, for the summary at the end of the run."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.issues: dict[str, list[str]] = defaultdict(list)

    def emit(self, record: logging.LogRecord) -> None:
        first_line = record.getMessage().splitlines()[0] if record.getMessage() else ""
        self.issues[getattr(record, "game", "run")].append(
            f"{record.levelname} {record.name}: {first_line}")


def _configure_logging(log_path: Path) -> tuple[_GameContext, _IssueCollector]:
    context = _GameContext()
    collector = _IssueCollector()
    collector.addFilter(context)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(context)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s [%(game)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))

    # The console shows this script's progress and every warning or error, not the play-by-play.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.addFilter(context)
    console.addFilter(lambda r: r.name == logger.name or r.levelno >= logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)-7s [%(game)s] %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in (file_handler, console, collector):
        root.addHandler(handler)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return context, collector


# arguments

def _parse_model(arg: str) -> SeatSpec:
    """Parse ``provider:model`` (split on the FIRST colon; model names contain colons). The local
    ``think`` flag comes from config.llm_models when the model is listed there."""
    if ":" not in arg:
        raise SystemExit(f"{arg!r} must be provider:model (e.g. ollama:llama3.1:8b).")
    provider, model_name = (part.strip() for part in arg.split(":", 1))
    provider = provider.lower()
    if provider not in _KNOWN_PROVIDERS or not model_name:
        raise SystemExit(f"{arg!r} must be provider:model, with provider one of {_KNOWN_PROVIDERS}.")
    configured = cast("dict[str, dict[str, Any]]", config.llm_models)
    for provider_key, models in configured.items():
        if provider_key.lower() == provider and model_name in models:
            return SeatSpec(provider=provider, model_name=model_name,
                            think=bool((models[model_name] or {}).get("think", False)))
    return SeatSpec(provider=provider, model_name=model_name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="play_games.py",
        description="Play N games between two models and log everything to a text file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-a", required=True, metavar="PROVIDER:MODEL",
                   help="the first model, e.g. ollama:llama3.1:8b.")
    p.add_argument("--model-b", metavar="PROVIDER:MODEL",
                   help="the second model; omit it to play model A against itself.")
    p.add_argument("--games", type=int, default=1, help="number of games (default 1).")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="sampling temperature (default 0.7).")
    p.add_argument("--master-seed", type=int,
                   help="master seed; every per-call seed derives from it. Default: from the clock "
                        "(logged, so a game can be replayed).")
    p.add_argument("--board-type", choices=("probe", "control"),
                   help="play only boards of this type (default: both).")
    p.add_argument("--board-id", action="append", metavar="ID",
                   help="play this board; repeat for several. The games cycle through them.")
    p.add_argument("--alternate-seats", action=argparse.BooleanOptionalAction, default=True,
                   help="swap the models' seats every other game (default: on).")
    p.add_argument("--enforce-digests", action="store_true",
                   help="abort if a local model's digest differs from config.EXPECTED_LOCAL_DIGESTS.")
    p.add_argument("--log-dir", default="logs", help="directory for the log file (default: logs).")
    return p


# boards

def _select_boards(board_ids: Optional[list[str]], board_type: Optional[str],
                   games: int) -> list[Board]:
    """The board of each game: the requested ids, or every probe/control board of the bank sorted
    by id, cycled through as many times as needed."""
    loader = BoardLoader(_BOARD_DATA_PATH)
    bank = {b.board_id: b for group in loader.boards.values() for b in group}
    if board_ids:
        unknown = [i for i in board_ids if i not in bank]
        if unknown:
            raise SystemExit(f"Unknown board id(s): {unknown}. Known: {sorted(bank)}")
        pool = [bank[i] for i in board_ids]
    else:
        pool = sorted((b for b in bank.values()
                       if b.type in ("probe", "control") and b.type == (board_type or b.type)),
                      key=lambda b: b.board_id)
    if not pool:
        raise SystemExit(f"No boards to play under {_BOARD_DATA_PATH!r}. Run from the repo root.")
    return [pool[i % len(pool)] for i in range(games)]


# run header

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<git {' '.join(args)} failed: {exc}>"


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return host if "://" in host else f"http://{host}"


def _ollama_get(path: str) -> dict:
    with urllib.request.urlopen(_ollama_host() + path, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _check_ollama(specs: list[SeatSpec]) -> None:
    """Fail fast if a local model cannot be served, and log what the server will play with."""
    local = sorted({s.model_name for s in specs if s.provider == "ollama"})
    if not local:
        return
    try:
        version = _ollama_get("/api/version").get("version")
        tags = {m["name"]: m.get("digest") for m in _ollama_get("/api/tags").get("models", [])}
    except OSError as exc:
        raise SystemExit(f"No Ollama server at {_ollama_host()} ({exc}). Start it first "
                         "(ollama serve) or set OLLAMA_HOST.")
    logger.info("ollama server %s, version %s", _ollama_host(), version)
    missing = [m for m in local if m not in tags]
    if missing:
        raise SystemExit(f"Ollama does not have {missing}; pull them first (ollama pull <model>).")
    for model in local:
        served = f"sha256:{tags[model]}" if tags[model] and ":" not in tags[model] else tags[model]
        expected = config.EXPECTED_LOCAL_DIGESTS.get(model)
        if expected is None:
            logger.info("model %s digest %s (not in config.EXPECTED_LOCAL_DIGESTS)", model, served)
        elif served == expected:
            logger.info("model %s digest %s (matches config)", model, served)
        else:
            logger.warning("model %s digest %s differs from config.EXPECTED_LOCAL_DIGESTS %s",
                           model, served, expected)


def _log_run_header(args: argparse.Namespace, specs: list[SeatSpec], boards: list[Board],
                    master_seed: int) -> None:
    logger.info("command: %s", " ".join(shlex.quote(a) for a in sys.argv))
    logger.info("git commit %s on %s", _git("rev-parse", "HEAD"), _git("rev-parse", "--abbrev-ref", "HEAD"))
    changes = _git("status", "--porcelain")
    logger.info("uncommitted changes: %s", "none" if not changes else "\n" + changes)
    logger.info("python %s on %s", platform.python_version(), platform.platform())
    logger.info("models: A=%s:%s B=%s:%s (think=%s/%s); seats alternate: %s",
                specs[0].provider, specs[0].model_name, specs[1].provider, specs[1].model_name,
                specs[0].think, specs[1].think, args.alternate_seats)
    logger.info("games=%s temperature=%s master_seed=%s boards=%s", args.games, args.temperature,
                master_seed, [b.board_id for b in boards])
    logger.info("ollama num_ctx=%s, answer cap=%s tokens", config.OLLAMA_NUM_CTX,
                config.LLM_MAX_OUTPUT_TOKENS)

    # The call log names the constant messages of each request by text_digest; print each template
    # once here under that id, so every prompt can be rebuilt from the log alone.
    service = LLMService(temperature=args.temperature)
    logger.debug("prompt template fingerprint %s", service.template_fingerprint())
    for name, text in service._loaded_template_texts().items():
        logger.debug("template %s id=%s\n--- template ---\n%s\n--- end of template ---",
                     name, text_digest(text), text)


# the game, as it unfolds

def _role(card: WordCard, seat: int) -> CardRole:
    return card.llm_perspective_role if seat == 0 else card.human_perspective_role


def _board_table(board: Board, *, with_state: bool) -> str:
    lines = [f"  {'id':>2}  {'word':<14} {'seat 0':<9} {'seat 1':<9} {'gender':<8}"
             + ("  state" if with_state else "")]
    for card in board.cards:
        line = (f"  {card.id:>2}  {card.text:<14} {_role(card, 0).value:<9} "
                f"{_role(card, 1).value:<9} {card.category or '-':<8}")
        if with_state:
            state = []
            if card.revealed:
                state.append("covered by an agent card")
            if card.time_marker_by:
                state.append("time token(s) from seat(s) " + ",".join(map(str, card.time_marker_by)))
            line += "  " + ("; ".join(state) or "-")
        lines.append(line)
    return "\n".join(lines)


def _pending_on_board(engine: CodenamesDuetEngine) -> list[int]:
    """pending_words as the board says it (§2): green on the OTHER seat's side, not covered."""
    cards = engine.state.board.cards
    return [sum(1 for c in cards if _role(c, 1 - seat) == CardRole.AGENT and not c.revealed)
            for seat in (0, 1)]


def check_invariants(engine: CodenamesDuetEngine) -> list[str]:
    """Every rule of the game state that must hold between two turns; one message per breach."""
    s = engine.state
    cards = s.board.cards
    problems = []

    spent = s.bystander_tokens + s.check_tokens + s.penalty_tokens
    if s.timer_tokens + spent != 9:
        problems.append(f"tokens do not add up to 9 (§12): reserve={s.timer_tokens} "
                        f"bystander={s.bystander_tokens} check={s.check_tokens} "
                        f"penalty={s.penalty_tokens}")
    if not 0 <= s.timer_tokens <= 9:
        problems.append(f"reserve out of range: {s.timer_tokens}")
    on_words = sum(len(c.time_marker_by) for c in cards)
    if on_words != s.bystander_tokens:
        problems.append(f"{on_words} time token(s) on words but bystander_tokens={s.bystander_tokens}")
    if s.pending_words != _pending_on_board(engine):
        problems.append(f"pending_words={s.pending_words} but the board says "
                        f"{_pending_on_board(engine)} (§2)")

    for c in cards:
        if len(c.time_marker_by) > 2 or len(set(c.time_marker_by)) != len(c.time_marker_by):
            problems.append(f"{c.text}: time tokens {c.time_marker_by} (at most one per direction)")
        for guesser in c.time_marker_by:
            if _role(c, 1 - guesser) != CardRole.CIVILIAN:
                problems.append(f"{c.text}: time token from seat {guesser}, but the word is "
                                f"{_role(c, 1 - guesser).value} for the clue giver (§6.4)")
        if c.revealed and CardRole.AGENT not in (_role(c, 0), _role(c, 1)):
            problems.append(f"{c.text}: covered by an agent card but green on neither side")

    visible = {c.text.lower() for c in cards if not c.is_covered}
    validator = set(engine.clue_validator.visible_words)
    if visible != validator:
        problems.append(f"visible words for clue validation differ from the board (§8.2): "
                        f"validator has {sorted(validator - visible)} extra, misses "
                        f"{sorted(visible - validator)}")

    if s.is_game_over:
        if s.result not in _RESULTS:
            problems.append(f"unknown result {s.result!r}")
        won = s.result in ("victory", "victory_sd")
        if won != (s.pending_words == [0, 0]):
            problems.append(f"result {s.result} with pending_words={s.pending_words}")
    else:
        phase = s.current_phase
        if s.timer_tokens == 0 and phase not in (GamePhase.SUDDEN_DEATH_LLM,
                                                 GamePhase.SUDDEN_DEATH_HUMAN):
            problems.append(f"the reserve is empty but the phase is {phase.value} (§7.3)")
        if phase in (GamePhase.GIVING_CLUE, GamePhase.GUESSING):
            if s.clue_giver == s.guesser:
                problems.append(f"seat {s.clue_giver} is both clue giver and guesser")
            if s.pending_words[s.guesser] == 0:
                problems.append(f"seat {s.guesser} guesses with nothing pending (§6.8)")
        if phase == GamePhase.SUDDEN_DEATH_LLM and s.pending_words[0] == 0:
            problems.append("SUDDEN_DEATH_LLM with nothing pending for seat 0")
        if phase == GamePhase.SUDDEN_DEATH_HUMAN and (s.pending_words[1] == 0
                                                      or s.pending_words[0] != 0):
            problems.append(f"SUDDEN_DEATH_HUMAN with pending_words={s.pending_words}")
    return problems


@dataclass
class _GameWatch:
    """The observer handed to run_single_game: logs the state after every turn and checks it."""

    engine: Optional[CodenamesDuetEngine] = None
    dispatches: int = 0
    violations: list[str] = field(default_factory=list)
    last_turn: int = 0

    def __call__(self, engine: CodenamesDuetEngine, recorder: GameRecorder) -> None:
        try:
            self.engine = engine
            s = engine.state
            covered = [c.text for c in s.board.cards if c.revealed]
            tokens = [f"{c.text}<-seat{','.join(map(str, c.time_marker_by))}"
                      for c in s.board.cards if c.time_marker_by]
            logger.debug(
                "state after %s dispatch(es): turn=%s phase=%s giver=%s guesser=%s reserve=%s "
                "bystander=%s check=%s penalty=%s pending=%s covered=%s time_tokens=%s",
                self.dispatches, s.turn_number, s.current_phase.value, s.clue_giver, s.guesser,
                s.timer_tokens, s.bystander_tokens, s.check_tokens, s.penalty_tokens,
                s.pending_words, covered, tokens)
            problems = check_invariants(engine)
            if s.turn_number < self.last_turn:
                problems.append(f"turn number went back from {self.last_turn} to {s.turn_number}")
            self.last_turn = s.turn_number
            for problem in problems:
                logger.error("INVARIANT BROKEN: %s", problem)
            self.violations.extend(problems)
            self.dispatches += 1
        except Exception:  # the observer must never end the game
            logger.exception("the state observer failed")


@dataclass
class _GameOutcome:
    index: int
    board_id: str
    seats: tuple[str, str]
    result: GameRunResult
    seconds: float
    watch: _GameWatch


def _check_rankings(engine: CodenamesDuetEngine) -> None:
    """The measurement rankings are record-only, so a malformed one never stops the game: warn
    about the words in them that are not on the board, and the words ranked twice."""
    s = engine.state
    rankings = [(f"turn {c.turn_number}", c.confidence_ranking)
                for c in list(s.clue_history) + ([s.current_clue] if s.current_clue else [])]
    if s.sudden_death is not None:
        rankings += [(f"sudden death, seat {seat}", ranking)
                     for seat, ranking in s.sudden_death.rankings_by_seat.items()]
    for where, ranking in rankings:
        if ranking is None:
            continue
        words = [item.word.upper() for item in ranking.rankings]
        off_board = [w for w in words if s.board.get_card_id_by_word(w) is None]
        repeated = sorted({w for w in words if words.count(w) > 1})
        if off_board or repeated:
            logger.warning("measurement ranking (%s): words not on the board %s, ranked twice %s",
                           where, off_board, repeated)


def _log_game_end(watch: _GameWatch) -> None:
    engine = watch.engine
    if engine is None:
        logger.error("no state to report: the game failed before its first turn")
        return
    _check_rankings(engine)
    s = engine.state
    logger.info("final board:\n%s", _board_table(s.board, with_state=True))
    clues = list(s.clue_history) + ([s.current_clue] if s.current_clue else [])
    lines = [f"  turn {c.turn_number}: seat {c.clue_giver} '{c.clue}' {c.count} "
             f"targets={c.targets}" + (f" INVALID: {c.invalid_reason}" if c.invalid_reason else "")
             for c in clues]
    logger.info("clues given (%s):\n%s", len(clues), "\n".join(lines) or "  none")


async def _play(args: argparse.Namespace, specs: list[SeatSpec], boards: list[Board],
                master_seed: int, context: _GameContext) -> list[_GameOutcome]:
    outcomes = []
    for index, board in enumerate(boards):
        seats = list(reversed(specs)) if args.alternate_seats and index % 2 else list(specs)
        context.label = f"g{index + 1:02d}"
        watch = _GameWatch()
        names = (f"{seats[0].provider}:{seats[0].model_name}",
                 f"{seats[1].provider}:{seats[1].model_name}")
        logger.info("game %s/%s on %s: seat 0 = %s, seat 1 = %s", index + 1, len(boards),
                    board.board_id, names[0], names[1])
        dilemma = board.dilemma
        logger.info("board %s (type=%s, category=%s%s):\n%s", board.board_id, board.type,
                    board.category,
                    f", dilemma: target={dilemma.target} stereotypical_bridge="
                    f"{dilemma.stereotypical_bridge} neutral_bridge={dilemma.neutral_bridge}"
                    if dilemma else "",
                    _board_table(board, with_state=False))
        started = time.monotonic()
        result = await run_single_game(
            board=board, seat_specs=seats, master_seed=master_seed, temperature=args.temperature,
            game_index=index, persist=False, enforce_digests=args.enforce_digests,
            on_dispatch=watch)
        seconds = time.monotonic() - started
        _log_game_end(watch)
        if result.status != "completed":
            logger.error("game %s ended in error: %s", index + 1, result.error)
        logger.info("game %s: %s, result=%s, %s turn(s), %.0fs, game_id=%s seed_game=%s",
                    index + 1, result.status, result.result,
                    watch.engine.state.turn_number if watch.engine else "?", seconds,
                    result.game_id, result.seed_game)
        outcomes.append(_GameOutcome(index + 1, board.board_id, names, result, seconds, watch))
    context.label = "run"
    return outcomes


def _log_summary(outcomes: list[_GameOutcome], collector: _IssueCollector, log_path: Path) -> int:
    lines = ["", "=" * 100, "SUMMARY", "=" * 100]
    for o in outcomes:
        s = o.watch.engine.state if o.watch.engine else None
        lines.append(
            f"g{o.index:02d} {o.board_id:<26} {o.seats[0]} vs {o.seats[1]}  {o.result.status:<9} "
            f"{str(o.result.result):<17} turns={s.turn_number if s else '?':<3} "
            f"reserve={s.timer_tokens if s else '?'} penalties={s.penalty_tokens if s else '?'} "
            f"{o.seconds:.0f}s")
        for issue in collector.issues.get(f"g{o.index:02d}", []):
            lines.append(f"      {issue}")
    for issue in collector.issues.get("run", []):
        lines.append(f"  run: {issue}")
    errored = sum(o.result.status != "completed" for o in outcomes)
    broken = sum(bool(o.watch.violations) for o in outcomes)
    issues = sum(len(v) for v in collector.issues.values())
    lines += ["-" * 100,
              f"games={len(outcomes)} completed={len(outcomes) - errored} errored={errored} "
              f"games_with_broken_invariants={broken} warnings_and_errors={issues}",
              f"log: {log_path}", "=" * 100]
    logger.info("\n".join(lines))
    return 1 if errored or broken else 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games < 1:
        raise SystemExit("--games must be at least 1.")
    specs = [_parse_model(args.model_a), _parse_model(args.model_b or args.model_a)]
    master_seed = args.master_seed if args.master_seed is not None else int(time.time())
    boards = _select_boards(args.board_id, args.board_type, args.games)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"games_{datetime.now():%Y%m%d_%H%M%S}.log"
    context, collector = _configure_logging(log_path)
    logger.info("logging to %s", log_path)

    _log_run_header(args, specs, boards, master_seed)
    _check_ollama(specs)
    outcomes = asyncio.run(_play(args, specs, boards, master_seed, context))
    return _log_summary(outcomes, collector, log_path)


if __name__ == "__main__":
    sys.exit(main())
