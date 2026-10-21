"""Record a golden game: play one game against the Ollama daemon and write down every call to it.

The game is played by run_single_game, as the batch plays it, with no database. Every call the
local clients make to the daemon is recorded with what they sent and what came back, and the game
as stored (backend/app/core/golden.py). tests/unit/test_golden_games.py then plays it again from
the record, and fails at the first request that differs or at a game that comes out otherwise.

Record the golden games again after a change to what the models are sent (a prompt template, a
response schema, a seed): the test names the first request that changed.

Environment: an Ollama server up (OLLAMA_HOST, default http://localhost:11434) with the models.

Example:
    python scripts/record_golden_game.py --model-a ollama:llama3.1:8b --board-id control-000 \\
        --master-seed 7 --out tests/fixtures/golden/llama_control.json \\
        --description "llama3.1 against itself on a control board"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.app import config
from backend.app.core import golden, provenance
from backend.app.core.game_runner import SeatSpec, default_client_factory, run_single_game
from backend.app.core.llm_service import LLMService
from backend.app.core.loader import BoardLoader


def _parse_model(arg: str) -> SeatSpec:
    """``ollama:<model>``, with the ``think`` flag config.llm_models gives the model."""
    provider, _, model_name = arg.partition(":")
    if provider.lower() != "ollama" or not model_name:
        raise SystemExit(f"{arg!r} must be ollama:<model>: golden games are recorded at Ollama.")
    think = bool(config.llm_models.get("Ollama", {}).get(model_name, {}).get("think", False))
    return SeatSpec(provider="ollama", model_name=model_name, think=think)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a golden game against Ollama.")
    parser.add_argument(
        "--model-a",
        required=True,
        type=_parse_model,
        metavar="ollama:MODEL",
        help="seat 0's model.",
    )
    parser.add_argument(
        "--model-b",
        type=_parse_model,
        metavar="ollama:MODEL",
        help="seat 1's model (default: model A).",
    )
    parser.add_argument("--board-id", required=True, help="a board of data/boards.")
    parser.add_argument("--master-seed", required=True, type=int)
    parser.add_argument("--game-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", required=True, type=Path, help="the fixture to write.")
    parser.add_argument("--description", required=True, help="what the game covers.")
    return parser


async def _record(args: argparse.Namespace) -> int:
    boards = {b.board_id: b for group in BoardLoader().boards.values() for b in group}
    if args.board_id not in boards:
        raise SystemExit(f"no board {args.board_id!r} in data/boards.")
    board = boards[args.board_id]
    specs = [args.model_a, args.model_b or args.model_a]
    service = LLMService(temperature=args.temperature)
    stand_ins: dict[int, golden.RecordingOllama] = {}

    def client_factory(seat: int, spec: SeatSpec):
        client = default_client_factory(seat, spec)
        client._client = stand_ins[seat] = golden.RecordingOllama(client._client)  # type: ignore[attr-defined]
        return client

    captured: dict[str, Any] = {}
    result = await run_single_game(
        board=board,
        seat_specs=specs,
        master_seed=args.master_seed,
        temperature=args.temperature,
        game_index=args.game_index,
        persist=False,
        client_factory=client_factory,
        service=service,
        on_dispatch=lambda engine, recorder: captured.update(recorder=recorder),
    )
    failed = [f"seat {seat}: {e}" for seat, s in sorted(stand_ins.items()) for e in s.failed]
    if failed:
        print(
            "a call to the daemon failed, so the game cannot be played again from the "
            "record; not written:\n  " + "\n  ".join(failed),
            file=sys.stderr,
        )
        return 1
    fixture = {
        "version": golden.FIXTURE_VERSION,
        "description": args.description,
        "recorded": {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_version": provenance.git_code_version(),
            "ollama": provenance.ollama_server_version(),
            "prompt_template_version": service.template_fingerprint(),
            "digests": {
                spec.model_name: provenance.resolve_ollama_digest(spec.model_name) for spec in specs
            },
        },
        "board": board.model_dump(mode="json"),
        "seats": [
            {"provider": s.provider, "model_name": s.model_name, "think": s.think} for s in specs
        ],
        "master_seed": args.master_seed,
        "game_index": args.game_index,
        "temperature": args.temperature,
        "calls": [stand_ins[0].calls, stand_ins[1].calls],
        "expected": {
            "status": result.status,
            "error_code": result.error_code,
            "result": result.result,
            "transcript": golden.transcript(captured["recorder"]),
        },
    }
    golden.save(args.out, fixture)
    turns = fixture["expected"]["transcript"]["turns"]
    print(
        f"{args.out}: {result.status}, {result.result}, {len(turns)} turn(s), "
        f"{sum(len(c) for c in fixture['calls'])} call(s) to the daemon"
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(_record(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
