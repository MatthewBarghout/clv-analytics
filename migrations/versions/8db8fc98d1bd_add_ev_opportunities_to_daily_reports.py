"""add_ev_opportunities_to_daily_reports

Revision ID: 8db8fc98d1bd
Revises: fb633588cc6d
Create Date: 2026-01-14 02:08:53.357980

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8db8fc98d1bd'
down_revision: Union[str, Sequence[str], None] = 'fb633588cc6d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add ev_opportunities field to daily_clv_reports table
    op.add_column(
        'daily_clv_reports',
        sa.Column('ev_opportunities', sa.dialects.postgresql.JSONB(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove ev_opportunities field
    op.drop_column('daily_clv_reports', 'ev_opportunities')
