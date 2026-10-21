"""Golden games: record real games at the Ollama daemon, and play them again from the record.

A golden game is a game played by ``run_single_game`` against a real Ollama daemon, recorded at the
daemon's door: every call the local client made (``show`` for the capability probe, ``chat`` for
each draw) with what it sent and what came back. Played again, the same client talks to a stand-in
for the daemon that serves the recorded answers in order and checks that every request is the one
recorded. The game must then come out as it did: the same clues, proposals, measurements, reveals,
redraws and result.

So a change that should not change the game (a refactor of the engine, the service, the conductor
or the client) is checked against games real models played; one that changes what the models are
sent (a prompt, a response schema, a seed) shows up as the first request that differs, and the
golden games must be recorded again (``scripts/record_golden_game.py``).

A request is recorded by its model, options and ``think`` flag, the SHA-256 of its messages and of
its response schema, and the text of its last message (the game state, for a readable diff).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ollama import ChatResponse, ShowResponse

from backend.app.db.recorder import GameRecorder

FIXTURE_VERSION = 1


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def request_record(kwargs: dict[str, Any]) -> dict[str, Any]:
    """What identifies a ``chat`` request: everything it sends, the long parts by their hash."""
    messages = kwargs["messages"]
    return {
        "model": kwargs["model"],
        "options": kwargs.get("options"),
        "think": kwargs.get("think"),
        "format_sha256": _sha256(kwargs.get("format")),
        "messages_sha256": _sha256(messages),
        "last_message": messages[-1]["content"],
    }


class RecordingOllama:
    """Stands between a local client and its ``ollama.AsyncClient``, and records every call."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.calls: list[dict[str, Any]] = []
        # A call that failed (a timeout, the daemon down) cannot be played again from the record.
        self.failed: list[str] = []

    async def show(self, model: str) -> Any:
        response = await self._inner.show(model)
        self.calls.append(
            {
                "call": "show",
                "model": model,
                "capabilities": [str(c) for c in response.capabilities or []],
            }
        )
        return response

    async def chat(self, **kwargs: Any) -> Any:
        try:
            response = await self._inner.chat(**kwargs)
        except Exception as e:
            self.failed.append(repr(e))
            raise
        self.calls.append(
            {
                "call": "chat",
                "request": request_record(kwargs),
                "response": json.loads(response.model_dump_json()),
            }
        )
        return response

    async def close(self) -> None:
        await self._inner.close()


class ReplayMismatch(AssertionError):
    """A request of the game played again is not the one recorded."""


class ReplayOllama:
    """Stands in for the Ollama daemon: serves the recorded calls in order, checking each request
    against the one recorded."""

    def __init__(self, calls: list[dict[str, Any]], seat: int):
        self._calls = list(calls)
        self._seat = seat
        self.served = 0

    def _next(self, kind: str) -> dict[str, Any]:
        if self.served >= len(self._calls):
            raise ReplayMismatch(
                f"seat {self._seat}: call {self.served + 1} ({kind}) was never recorded"
            )
        call = self._calls[self.served]
        self.served += 1
        if call["call"] != kind:
            raise ReplayMismatch(
                f"seat {self._seat}: call {self.served} is a {kind}, the record has a "
                f"{call['call']}"
            )
        return call

    async def show(self, model: str) -> ShowResponse:
        call = self._next("show")
        # Built without validation: the client reads only the capabilities, and a ShowResponse
        # that failed to validate would make it fall back, in silence, to a model that cannot
        # think.
        return ShowResponse.model_construct(capabilities=call["capabilities"])

    async def chat(self, **kwargs: Any) -> ChatResponse:
        call = self._next("chat")
        sent, recorded = request_record(kwargs), call["request"]
        differing = sorted(key for key in recorded if sent.get(key) != recorded[key])
        if differing:
            detail = ""
            if sent["last_message"] != recorded["last_message"]:
                detail = (
                    f"\n--- recorded last message ---\n{recorded['last_message']}"
                    f"\n--- sent last message ---\n{sent['last_message']}"
                )
            raise ReplayMismatch(
                f"seat {self._seat}: request {self.served} differs from the recorded one in "
                f"{differing}; a change to what the models are sent needs the golden games "
                f"recorded again (scripts/record_golden_game.py).{detail}"
            )
        return ChatResponse.model_validate(call["response"])

    async def close(self) -> None:
        pass

    def unserved(self) -> int:
        return len(self._calls) - self.served


def _call(record: Any) -> dict[str, Any]:
    payload = record.raw_payload or {}
    return {
        "role": record.role,
        "requested_seed": record.requested_seed,
        "finish_reason": record.finish_reason,
        "redraws": {
            key: value for key, value in sorted(payload.items()) if key.endswith("_resamples")
        },
    }


def _proposal(proposal: Any) -> dict[str, Any] | None:
    if proposal is None:
        return None
    return {
        "proposals": list(proposal.proposals),
        "confidence": list(proposal.confidence),
        "reasoning": proposal.reasoning,
        "stop_reason": proposal.stop_reason,
        "call": _call(proposal.llm_call) if proposal.llm_call else None,
    }


def _ranking(ranking: Any) -> dict[str, Any] | None:
    if ranking is None:
        return None
    return {
        "rankings": [[item.word, item.confidence] for item in ranking.rankings],
        "reasoning": ranking.reasoning,
        "call": _call(ranking.llm_call) if ranking.llm_call else None,
    }


def transcript(recorder: GameRecorder) -> dict[str, Any]:
    """What the game stores, without what a replay does not reproduce (prompts, raw payloads,
    latencies): every turn's clue, proposals, measurements and reveals, the calls behind them
    with their seeds and redraws, and the outcome."""
    turns = []
    for turn in recorder.turns:
        clue = turn.clue
        turns.append(
            {
                "turn_number": turn.turn_number,
                "phase": turn.phase,
                "clue_giver_seat": turn.clue_giver_seat,
                "clue": None
                if clue is None
                else {
                    "clue_word": clue.clue_word,
                    "count": clue.count,
                    "reasoning": clue.reasoning,
                    "invalid_reason": clue.invalid_reason,
                    "targets_raw": list(clue.targets_raw),
                    "targets_resolved": [
                        [
                            t.word,
                            t.card_id,
                            getattr(t.giver_role, "value", None),
                            t.revealed_at_clue,
                        ]
                        for t in clue.targets_resolved
                    ],
                    "calls": [_call(c) for c in clue.llm_calls],
                },
                "play": _proposal(turn.play_proposal),
                "measurement": _ranking(turn.measurement),
                "sd_play": {
                    str(seat): _proposal(p) for seat, p in sorted(turn.sd_play_by_seat.items())
                },
                "sd_measurement": {
                    str(seat): _ranking(r)
                    for seat, r in sorted(turn.sd_measurement_by_seat.items())
                },
                "reveals": [
                    [
                        r.card_id,
                        r.result_role,
                        r.acting_seat,
                        r.timer_tokens_after,
                        r.ended_turn,
                        r.ended_game,
                        r.proposal_index,
                    ]
                    for r in turn.reveals
                ],
            }
        )
    return {
        "start_player": recorder.start_player,
        "result": recorder.result,
        "timer_tokens_final": recorder.timer_tokens_final,
        "turns": turns,
    }


def load(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("version") != FIXTURE_VERSION:
        raise ValueError(
            f"{path}: golden game version {fixture.get('version')}, expected {FIXTURE_VERSION}"
        )
    return fixture


def save(path: Path, fixture: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
