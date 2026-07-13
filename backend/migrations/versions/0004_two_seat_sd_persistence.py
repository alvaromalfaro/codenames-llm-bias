"""two-seat sudden death: widen guess_proposal unique key with guesser_seat

Revision ID: 0004_two_seat_sd_persistence
Revises: 0003_game_write_path
Create Date: 2026-07-13

Additive, reversible change enabling two-seat sudden-death persistence:

Sudden death is a single collective turn on which both seats emit a play proposal and a
confidence-ranking measurement under the same ``turn_id``. The pre-existing ``UNIQUE (turn_id, kind)`` 
forbids that (two ``'play'`` rows, or two ``'measurement'`` rows, on one turn collide). This widens 
the key to ``UNIQUE (turn_id, kind, guesser_seat)`` - a strict->looser change, safe for existing 
rows - so both seats coexist while still forbidding a duplicate of the same kind for the same seat. 
``guess_proposal.guesser_seat`` already exists; only the constraint changes.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_two_seat_sd_persistence"
down_revision: Union[str, None] = "0003_game_write_path"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_guess_proposal_turn_kind", "guess_proposal", type_="unique"
    )
    op.create_unique_constraint(
        "uq_guess_proposal_turn_kind_seat",
        "guess_proposal",
        ["turn_id", "kind", "guesser_seat"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_guess_proposal_turn_kind_seat", "guess_proposal", type_="unique"
    )
    op.create_unique_constraint(
        "uq_guess_proposal_turn_kind", "guess_proposal", ["turn_id", "kind"]
    )
