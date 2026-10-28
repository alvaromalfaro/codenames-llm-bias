"""Runs read back from the database: what ``scripts/manage_runs.py`` lists and shows, and what a
resumed batch checks.

A run is one batch: the provenance it was minted with, the notes it gathered (each resume appends
one), and its games. Reading writes nothing.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.models import GameModel, RunModel
from backend.app.db.session import session_scope


@dataclass(frozen=True)
class RunSummary:
    """A run as it was minted, its notes, and how many of its games are recorded under each
    ``game_status``."""

    run_id: str
    created_at: datetime
    master_seed: Optional[int]
    temperature: float
    code_version: Optional[str]
    prompt_template_version: Optional[str]
    model_registry_snapshot: Optional[dict]
    stimulus_manifest: Optional[dict]
    environment_snapshot: Optional[dict]
    notes: Optional[str]
    games_by_status: dict[str, int]

    @property
    def games(self) -> int:
        return sum(self.games_by_status.values())


def _summaries(session: Session, runs: Sequence[RunModel]) -> list[RunSummary]:
    """Summarise ``runs``, in their order, counting their games in one grouped query."""
    # game.run_id is nullable, though none of these games has it NULL (they are the runs').
    counts: dict[str | None, dict[str, int]] = defaultdict(dict)
    if runs:
        rows = session.execute(
            select(GameModel.run_id, GameModel.game_status, func.count())
            .where(GameModel.run_id.in_([run.id for run in runs]))
            .group_by(GameModel.run_id, GameModel.game_status)
        ).all()
        for run_id, status, n in rows:
            counts[run_id][status] = int(n)
    return [
        RunSummary(
            run_id=run.id,
            created_at=run.created_at,
            master_seed=None if run.master_seed is None else int(run.master_seed),
            temperature=run.temperature,
            code_version=run.code_version,
            prompt_template_version=run.prompt_template_version,
            model_registry_snapshot=run.model_registry_snapshot,
            stimulus_manifest=run.stimulus_manifest,
            environment_snapshot=run.environment_snapshot,
            notes=run.notes,
            games_by_status=dict(sorted(counts[run.id].items())),
        )
        for run in runs
    ]


def list_runs() -> list[RunSummary]:
    """Every run, oldest first."""
    with session_scope() as session:
        runs = (
            session.execute(select(RunModel).order_by(RunModel.created_at, RunModel.id))
            .scalars()
            .all()
        )
        return _summaries(session, runs)


def get_run(run_id: str) -> Optional[RunSummary]:
    """The run ``run_id``, or None when there is no such run (or ``run_id`` is not a run id at
    all)."""
    try:
        uuid.UUID(run_id)
    except ValueError:
        return None
    with session_scope() as session:
        run = session.get(RunModel, run_id)
        return None if run is None else _summaries(session, [run])[0]
