"""add daily clv reports table

Revision ID: 852e3163ab49
Revises: 0fe52f00493f
Create Date: 2026-01-06 21:14:51.325840

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = '852e3163ab49'
down_revision: Union[str, Sequence[str], None] = '0fe52f00493f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'daily_clv_reports',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('report_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('games_analyzed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_opportunities', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_clv', sa.Float(), nullable=True),
        sa.Column('median_clv', sa.Float(), nullable=True),
        sa.Column('positive_clv_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('positive_clv_percentage', sa.Float(), nullable=True),
        sa.Column('best_opportunities', JSONB, nullable=True),
        sa.Column('by_bookmaker', JSONB, nullable=True),
        sa.Column('by_market', JSONB, nullable=True),
        sa.Column('game_summaries', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_daily_clv_reports_date', 'daily_clv_reports', ['report_date'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_daily_clv_reports_date', table_name='daily_clv_reports')
    op.drop_table('daily_clv_reports')
