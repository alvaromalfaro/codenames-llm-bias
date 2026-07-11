"""game write-path: guess_proposal.kind and widened llm_call.role

Revision ID: 0003_game_write_path
Revises: 0002_prompt_capture
Create Date: 2026-07-11

Additive, reversible change supporting the per-game write-path:

- Adds ``guess_proposal.kind`` (``'play'`` | ``'measurement'``) so a single turn can carry both the
  guesser's play proposal and the out-of-band confidence-ranking measurement. The pre-existing
  single-column ``UNIQUE (turn_id)`` is dropped and replaced by ``UNIQUE (turn_id, kind)`` so both
  kinds coexist on one turn while still forbidding duplicates of the same kind.
- Widens the ``llm_call.role`` CHECK to admit the two measurement roles (``'measurement'``, 
  ``'measurement_sd'``) alongside the existing play roles.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_game_write_path"
down_revision: Union[str, None] = "0002_prompt_capture"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # guess_proposal.kind (temporary default backfills any existing rows, then removed).
    op.add_column(
        "guess_proposal",
        sa.Column(
            "kind", sa.Text(), nullable=False, server_default=sa.text("'play'")
        ),
    )
    op.alter_column("guess_proposal", "kind", server_default=None)
    op.create_check_constraint(
        "ck_guess_proposal_kind",
        "guess_proposal",
        "kind IN ('play','measurement')",
    )
    # Replace the single-column UNIQUE(turn_id) with UNIQUE(turn_id, kind).
    op.drop_constraint(
        "guess_proposal_turn_id_key", "guess_proposal", type_="unique"
    )
    op.create_unique_constraint(
        "uq_guess_proposal_turn_kind", "guess_proposal", ["turn_id", "kind"]
    )

    # Widen the llm_call.role CHECK to admit the measurement roles.
    op.drop_constraint("ck_llm_call_role", "llm_call", type_="check")
    op.create_check_constraint(
        "ck_llm_call_role",
        "llm_call",
        "role IN ('clue_giver','guesser','guesser_sd','measurement','measurement_sd')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_llm_call_role", "llm_call", type_="check")
    op.create_check_constraint(
        "ck_llm_call_role",
        "llm_call",
        "role IN ('clue_giver','guesser','guesser_sd')",
    )

    op.drop_constraint(
        "uq_guess_proposal_turn_kind", "guess_proposal", type_="unique"
    )
    op.create_unique_constraint(
        "guess_proposal_turn_id_key", "guess_proposal", ["turn_id"]
    )
    op.drop_constraint(
        "ck_guess_proposal_kind", "guess_proposal", type_="check"
    )
    op.drop_column("guess_proposal", "kind")
