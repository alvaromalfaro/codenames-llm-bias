"""prompt capture columns

Revision ID: 0002_prompt_capture
Revises: 0001_initial
Create Date: 2026-07-10

Additive, schema-only change: adds ``llm_call.rendered_prompt`` (the full messages array as sent to
the provider) and ``run.prompt_template_version`` (a version/hash of the prompt-template set used 
for the batch). Both are nullable; nothing writes to them yet.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_prompt_capture"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_call",
        sa.Column("rendered_prompt", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "run",
        sa.Column("prompt_template_version", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("run", "prompt_template_version")
    op.drop_column("llm_call", "rendered_prompt")
