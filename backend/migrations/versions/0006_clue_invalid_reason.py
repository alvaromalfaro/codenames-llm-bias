"""invalid clues: clue.invalid_reason

Revision ID: 0006_clue_invalid_reason
Revises: 0005_run_delete_cascade
Create Date: 2026-10-07

Additive, reversible change: adds ``clue.invalid_reason``, why the clue broke the validity rules, NULL
for a valid clue. Invalid clues used to be rejected (the model regenerated them, the human retried);
they are now played as the Duet rules say, with a penalty token (§8.4), so the clue row itself has to
say whether it was invalid. Nullable, so the rows already recorded read as valid clues, which they
are: an invalid clue never reached the clue table before.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006_clue_invalid_reason"
down_revision: Union[str, None] = "0005_run_delete_cascade"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "clue",
        sa.Column("invalid_reason", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("clue", "invalid_reason")
