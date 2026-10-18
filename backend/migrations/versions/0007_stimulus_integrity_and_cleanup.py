"""stimulus integrity, run provenance, game errors, indexes; drop the columns never written

Revision ID: 0007_stimulus_integrity_and_cleanup
Revises: 0006_clue_invalid_reason
Create Date: 2026-10-18

Adds, all nullable, so the rows already recorded stay valid:
  * ``board.content_sha256``: the SHA-256 of the board artifact's canonical JSON, so a board whose
    file changed under the same ``board_id`` is caught instead of played in silence;
  * ``run.stimulus_manifest``: every scheduled board's ``board_id`` with that hash, so a resumed
    batch, and the analysis, can tell the run played the boards it was minted with;
  * ``run.environment_snapshot``: the environment the batch ran in (interpreter, packages, server
    versions, the clue validator's data);
  * ``game.error_code`` and ``game.error_message``: why a game recorded as an error failed.

Indexes the foreign keys the batch, the cascade and the analysis filter on: ``game.run_id``,
``game.board_id``, ``guess_proposal_item.reveal_event_id`` (walked by its ON DELETE SET NULL), and
``llm_call.role``.

Drops the columns nothing ever wrote, whose defaults a reader could take for data:
``reveal_event.time_marker_placed`` (always false, even when a civilian took a token),
``llm_call.execution_mode``, ``run.regime_label``, ``game_seat.precision`` and
``game_seat.requested_seed`` (each call's seed is ``llm_call.requested_seed``). ``run.notes`` stays:
each resume of a batch appends a line to it.

The downgrade restores the dropped columns empty (with their defaults), as they always were.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0007_stimulus_integrity_and_cleanup"
down_revision: Union[str, None] = "0006_clue_invalid_reason"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEXES = (
    ("ix_game_run_id", "game", ["run_id"]),
    ("ix_game_board_id", "game", ["board_id"]),
    ("ix_guess_proposal_item_reveal_event_id", "guess_proposal_item", ["reveal_event_id"]),
    ("ix_llm_call_role", "llm_call", ["role"]),
)


def upgrade() -> None:
    op.add_column("board", sa.Column("content_sha256", sa.Text(), nullable=True))
    op.add_column("run", sa.Column("stimulus_manifest", postgresql.JSONB(), nullable=True))
    op.add_column("run", sa.Column("environment_snapshot", postgresql.JSONB(), nullable=True))
    op.add_column("game", sa.Column("error_code", sa.Text(), nullable=True))
    op.add_column("game", sa.Column("error_message", sa.Text(), nullable=True))

    for name, table, columns in _INDEXES:
        op.create_index(name, table, columns)

    op.drop_column("reveal_event", "time_marker_placed")
    op.drop_column("llm_call", "execution_mode")
    op.drop_column("run", "regime_label")
    op.drop_column("game_seat", "precision")
    op.drop_column("game_seat", "requested_seed")


def downgrade() -> None:
    op.add_column("game_seat", sa.Column("requested_seed", sa.Numeric(20, 0), nullable=True))
    op.add_column("game_seat", sa.Column("precision", sa.Text(), nullable=True))
    op.add_column("run", sa.Column("regime_label", sa.Text(), nullable=True))
    op.add_column("llm_call", sa.Column("execution_mode", sa.Text(), nullable=True))
    op.add_column("reveal_event", sa.Column(
        "time_marker_placed", sa.Boolean(), nullable=False, server_default=sa.text("false")))

    for name, table, _ in reversed(_INDEXES):
        op.drop_index(name, table_name=table)

    op.drop_column("game", "error_message")
    op.drop_column("game", "error_code")
    op.drop_column("run", "environment_snapshot")
    op.drop_column("run", "stimulus_manifest")
    op.drop_column("board", "content_sha256")
