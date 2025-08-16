import asyncio
from collections import deque
from datetime import datetime
from typing import Deque, AsyncIterator

from .config import settings


class LogBus:
	def __init__(self, capacity: int | None = None) -> None:
		self.capacity = capacity or settings.log_buffer_size
		self.buffer: Deque[str] = deque(maxlen=self.capacity)
		self._queue: asyncio.Queue[str] = asyncio.Queue()

	def push(self, line: str) -> None:
		timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		formatted_line = f"[{timestamp}] {line}"
		self.buffer.append(formatted_line)
		self._queue.put_nowait(formatted_line)

	def tail(self, limit: int = 200) -> list[str]:
		return list(self.buffer)[-limit:]

	async def stream(self) -> AsyncIterator[str]:
		while True:
			line = await self._queue.get()
			yield line


log_bus = LogBus()
