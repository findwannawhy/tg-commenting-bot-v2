from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Text, select
from datetime import datetime

from .config import settings


class Base(DeclarativeBase):
	pass


class Channel(Base):
	__tablename__ = "channels"
	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	url: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
	note: Mapped[str] = mapped_column(String(255), default="")
	enabled: Mapped[int] = mapped_column(Integer, default=1)
	added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
	channel_name: Mapped[str] = mapped_column(String(255), default="")


class KVSetting(Base):
	__tablename__ = "settings"
	key: Mapped[str] = mapped_column(String(100), primary_key=True)
	value: Mapped[str | None] = mapped_column(Text, nullable=True)


class State(Base):
	__tablename__ = "state"
	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	is_running: Mapped[int] = mapped_column(Integer, default=0)
	is_dry_run: Mapped[int] = mapped_column(Integer, default=0)
	cool_down_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


def make_engine() -> AsyncEngine:
	return create_async_engine(
		settings.database_url,
		future=True,
		pool_pre_ping=True,
		connect_args={"timeout": 15},
	)


engine: AsyncEngine = make_engine()


async_session_factory = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
	async with async_session_factory() as session:
		yield session


async def init_db() -> None:
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
		# lightweight migration: ensure channels.channel_name exists
		try:
			res = await conn.exec_driver_sql("PRAGMA table_info(channels)")
			cols = [row[1] for row in res.fetchall()]  # [cid, name, type, notnull, dflt_value, pk]
			if "channel_name" not in cols:
				await conn.exec_driver_sql("ALTER TABLE channels ADD COLUMN channel_name VARCHAR(255) DEFAULT ''")
		except Exception:
			pass
	# ensure state row exists
	async with async_session_factory() as session:
		res = await session.execute(select(State).where(State.id == 1))
		row = res.scalar_one_or_none()
		if not row:
			session.add(State(id=1, is_running=0, is_dry_run=0))
			await session.commit()
