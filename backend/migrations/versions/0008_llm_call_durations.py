"""llm_call: the server's breakdown of a call's latency

Revision ID: 0008_llm_call_durations
Revises: 0007_stimulus_and_cleanup
Create Date: 2026-10-22

Adds, nullable, ``llm_call.load_ms``, ``prompt_eval_ms`` and ``eval_ms``: what ollama reports a
call spent loading the model, reading the prompt and writing the answer, in milliseconds.
``latency_ms`` is their sum (with some overhead), so a reload, which happens at every turn when the
two models of a pairing do not fit in VRAM together, could not be told apart from a slow answer.

The rows already recorded are filled from their ``raw_payload``, which holds ollama's response as
it came, in nanoseconds. An API call reports no such breakdown, and keeps them NULL.

The downgrade drops them; the durations stay in ``raw_payload``.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008_llm_call_durations"
down_revision: Union[str, None] = "0007_stimulus_and_cleanup"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Each column, with the key of ollama's response it is read from.
_DURATIONS = (
    ("load_ms", "load_duration"),
    ("prompt_eval_ms", "prompt_eval_duration"),
    ("eval_ms", "eval_duration"),
)

# Fills the columns of the rows recorded before them: those of an ollama call, whose raw_payload
# carries total_duration. A key ollama left out stays NULL.
BACKFILL = (
    "UPDATE llm_call SET "
    + ", ".join(
        f"{column} = round((raw_payload->>'{key}')::numeric / 1000000)"
        for column, key in _DURATIONS
    )
    + " WHERE raw_payload->>'total_duration' IS NOT NULL"
)


def upgrade() -> None:
    for column, _ in _DURATIONS:
        op.add_column("llm_call", sa.Column(column, sa.Integer(), nullable=True))
    op.execute(BACKFILL)


def downgrade() -> None:
    for column, _ in reversed(_DURATIONS):
        op.drop_column("llm_call", column)
