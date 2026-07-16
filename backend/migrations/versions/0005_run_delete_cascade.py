"""run deletion: cascade game.run_id and set-null the intra-turn reveal back-reference

Revision ID: 0005_run_delete_cascade
Revises: 0004_two_seat_sd_persistence
Create Date: 2026-07-16

Additive, reversible change enabling a run (and its whole game subtree) to be deleted, which the
deterministic (master_seed, game_index) game identity now requires: re-running an experiment collides
on the game.id primary key, so a re-run must first delete the run. Two foreign keys block that today:

1. ``game.run_id`` -> ``run.id`` had no ``ondelete``, so deleting a run that owns any game raised
   ``game_run_id_fkey``. It becomes ``ON DELETE CASCADE`` so the run tear-down reaches its games (and,
   through the already-cascading game/turn subtree, everything below them).

2. ``guess_proposal_item.reveal_event_id`` -> ``reveal_event.id`` had no ``ondelete``. Both are
   deleted together when a turn/game/run is torn down, but Postgres deletes ``reveal_event`` before
   the referencing ``guess_proposal_item`` and the default restrict raised
   ``guess_proposal_item_reveal_event_id_fkey`` mid-cascade. ``ON DELETE SET NULL`` removes that
   ordering constraint (the item is itself deleted moments later by its own parent cascade, so the
   NULL is transient; the column is already nullable). The two sibling back-references
   ``clue.llm_call_id`` and ``guess_proposal.llm_call_id`` were probed and tore down cleanly, so they
   are deliberately left untouched.

Both FKs are unnamed in the schema, so Postgres derived ``game_run_id_fkey`` /
``guess_proposal_item_reveal_event_id_fkey``; recreating with ``constraint_name=None`` keeps them
unnamed (Postgres re-derives the same names), matching the convention that every FK in this schema is
unnamed and keeping ``alembic check`` clean against models.py.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005_run_delete_cascade"
down_revision: Union[str, None] = "0004_two_seat_sd_persistence"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("game_run_id_fkey", "game", type_="foreignkey")
    op.create_foreign_key(
        None, "game", "run", ["run_id"], ["id"], ondelete="CASCADE"
    )

    op.drop_constraint(
        "guess_proposal_item_reveal_event_id_fkey", "guess_proposal_item",
        type_="foreignkey",
    )
    op.create_foreign_key(
        None, "guess_proposal_item", "reveal_event", [
            "reveal_event_id"], ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "guess_proposal_item_reveal_event_id_fkey", "guess_proposal_item",
        type_="foreignkey",
    )
    op.create_foreign_key(
        None, "guess_proposal_item", "reveal_event", [
            "reveal_event_id"], ["id"]
    )

    op.drop_constraint("game_run_id_fkey", "game", type_="foreignkey")
    op.create_foreign_key(None, "game", "run", ["run_id"], ["id"])
