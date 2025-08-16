from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Any


class Metrics:
	def __init__(self) -> None:
		self.counters: Dict[str, int] = {
			"processed_posts_total": 0,
			"comments_sent_total": 0,
			"skipped_total": 0,
			"skipped_ads_total": 0,
			"skipped_heuristic_total": 0,
			"skipped_threshold_total": 0,
			"floodwait_events_total": 0,
			"errors_total": 0,
		}
		self.llm_lat_ms: Deque[float] = deque(maxlen=500)
		self.tg_send_lat_ms: Deque[float] = deque(maxlen=500)

	def inc(self, name: str, value: int = 1) -> None:
		self.counters[name] = self.counters.get(name, 0) + value

	def reset(self) -> None:
		for k in list(self.counters.keys()):
			self.counters[k] = 0
		self.llm_lat_ms.clear()
		self.tg_send_lat_ms.clear()

	def observe_llm(self, ms: float) -> None:
		self.llm_lat_ms.append(ms)

	def observe_tg_send(self, ms: float) -> None:
		self.tg_send_lat_ms.append(ms)

	def snapshot(self) -> Dict[str, Any]:
		def pct(lst: Deque[float], p: float) -> float:
			if not lst:
				return 0.0
			s = sorted(lst)
			k = int((len(s) - 1) * p)
			return s[k]
		return {
			**self.counters,
			"llm_latency_ms_p50": pct(self.llm_lat_ms, 0.50),
			"llm_latency_ms_p95": pct(self.llm_lat_ms, 0.95),
			"tg_send_latency_ms_p50": pct(self.tg_send_lat_ms, 0.50),
			"tg_send_latency_ms_p95": pct(self.tg_send_lat_ms, 0.95),
		}

	def time_llm(self):
		start = time.perf_counter()
		class Ctx:
			def __enter__(_self):
				return None
			def __exit__(_self, exc_type, exc, tb):
				elapsed = (time.perf_counter() - start) * 1000.0
				self.observe_llm(elapsed)
		return Ctx()

	def time_tg(self):
		start = time.perf_counter()
		class Ctx:
			def __enter__(_self):
				return None
			def __exit__(_self, exc_type, exc, tb):
				elapsed = (time.perf_counter() - start) * 1000.0
				self.observe_tg_send(elapsed)
		return Ctx()


metrics = Metrics()
