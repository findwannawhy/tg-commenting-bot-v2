from __future__ import annotations

import asyncio
import base64
import os
import shutil
import random
from datetime import datetime, timedelta
from collections import deque
from typing import Optional
import string

from telethon import TelegramClient, events, errors
from telethon.tl.functions.channels import GetFullChannelRequest, JoinChannelRequest, GetParticipantRequest
from telethon.tl.functions.messages import GetDiscussionMessageRequest
from telethon.tl.types import PeerChannel

from .config import settings
from .db import get_session, Channel
from .logging_bus import log_bus
from .ad_detector import fast_heuristic_is_ad
from .ai_service import AIService
from .metrics import metrics
from sqlalchemy import select
from urllib.parse import urlparse


class BotRunner:
	def __init__(self) -> None:
		self.loop = asyncio.get_event_loop()
		self.client: Optional[TelegramClient] = None
		self.ai: Optional[AIService] = None
		self.running = False
		self.channel_to_discussion: dict[int, int] = {}
		self.cool_down_until: Optional[datetime] = None
		# per-channel counters how many posts to skip before next comment
		self.skip_counters: dict[int, int] = {}
		self._auth_future: Optional[asyncio.Future[str]] = None
		self._awaiting: Optional[str] = None  # "code" or "password"
		self._join_attempts = deque()
		self._last_join_at: Optional[datetime] = None
		self._processed_grouped_ids = deque(maxlen=100)
		self._post_counter: int = 0

	async def start(self) -> None:
		if self.running:
			return
		start_ts = datetime.utcnow()
		log_bus.push("[runner] START: refreshing AI settings…")
		await self._refresh_ai_from_settings()
		log_bus.push("[runner] START: ensuring Telegram client…")
		await self._ensure_client()
		log_bus.push("[runner] START: loading subscriptions…")
		await self._load_subscriptions()
		self.running = True
		dur = (datetime.utcnow() - start_ts).total_seconds()
		log_bus.push(f"[runner] START: done in {dur:.1f}s")

	async def stop(self) -> None:
		log_bus.push("[runner] STOP: begin")
		self.running = False
		if self.client and self.client.is_connected():
			await self.client.disconnect()
		log_bus.push("[runner] STOP: done")

	async def restart(self) -> None:
		log_bus.push("[runner] RESTART: begin")
		await self.stop()
		await self.start()
		log_bus.push("[runner] RESTART: done")

	async def reset_session(self) -> None:
		# disconnect client and clear volatile state
		try:
			if self.client and self.client.is_connected():
				await self.client.disconnect()
		except Exception:
			pass
		self.client = None
		self.running = False
		self.channel_to_discussion.clear()
		self.skip_counters.clear()
		self.cool_down_until = None
		# remove sessions directory and recreate it empty
		sessions_dir = settings.sessions_dir
		try:
			shutil.rmtree(sessions_dir, ignore_errors=True)
		except Exception:
			pass
		os.makedirs(sessions_dir, exist_ok=True)
		try:
			os.chmod(sessions_dir, 0o700)
		except Exception:
			pass
		log_bus.push("[auth] Session reset. Please enter a new phone number or press Start/Restart for a new authorization.")

	async def _ensure_client(self) -> None:
		sessions_dir = settings.sessions_dir
		os.makedirs(sessions_dir, exist_ok=True)
		try:
			os.chmod(sessions_dir, 0o700)
		except Exception:
			pass
		session_path = os.path.join(sessions_dir, "account")

		log_bus.push("[client] Resolving proxy and credentials…")
		proxy_url = await self._get_proxy_url()
		proxy = None
		if proxy_url:
			p = urlparse(proxy_url)
			try:
				import socks
				scheme_map = {
					"socks5": socks.SOCKS5,
					"socks5h": socks.SOCKS5,
					"http": socks.HTTP,
					"https": socks.HTTP,
				}
				ptype = scheme_map.get(p.scheme)
				if ptype:
					proxy = (ptype, p.hostname, p.port, True, p.username, p.password)
			except Exception:
				proxy = None

		# allow overrides from DB for telegram creds
		from .db import get_session, KVSetting
		api_id: Optional[int] = None
		api_hash: Optional[str] = None
		async with get_session() as session:
			api_id_db = await session.scalar(select(KVSetting.value).where(KVSetting.key == "telegram_api_id"))
			api_hash_db = await session.scalar(select(KVSetting.value).where(KVSetting.key == "telegram_api_hash"))
			if api_id_db:
				try:
					api_id = int(api_id_db)
				except Exception:
					api_id = None
			if api_hash_db:
				api_hash = api_hash_db
		if not api_id or not api_hash:
			log_bus.push("[auth] TELEGRAM_API_ID/TELEGRAM_API_HASH not set. Please specify them on the Settings page.")
			raise RuntimeError("TELEGRAM API credentials not configured")

		# basic validation to fail fast with clear message
		if api_id <= 0:
			log_bus.push("[auth] TELEGRAM_API_ID must be a positive integer.")
			raise RuntimeError("Invalid TELEGRAM_API_ID")
		ah = api_hash.strip()
		if not (len(ah) == 32 and all(ch in string.hexdigits for ch in ah)):
			log_bus.push("[auth] TELEGRAM_API_HASH seems incorrect (a 32-character hex string is expected). Check the value at my.telegram.org → API development tools.")
			raise RuntimeError("Invalid TELEGRAM_API_HASH format")

		self.client = TelegramClient(session_path, api_id, api_hash, proxy=proxy)
		log_bus.push("[client] Connecting…")
		await self.client.connect()
		if not await self.client.is_user_authorized():
			log_bus.push("[client] Not authorized → login flow…")
			await self._login_flow()
		else:
			log_bus.push("[client] Already authorized.")

	async def _refresh_ai_from_settings(self) -> None:
		# read OPENAI_API_KEY and PROXY_URL overrides from DB
		from .db import get_session, KVSetting
		api_key: Optional[str] = None
		proxy_url: Optional[str] = None
		async with get_session() as session:
			api_key_db = await session.scalar(select(KVSetting.value).where(KVSetting.key == "openai_api_key"))
			proxy_url_db = await session.scalar(select(KVSetting.value).where(KVSetting.key == "proxy_url"))
			if api_key_db:
				api_key = api_key_db
			if proxy_url_db:
				proxy_url = proxy_url_db
		self.ai = AIService(api_key, settings.model_default, proxy_url=proxy_url) if api_key else None

	async def _login_flow(self) -> None:
		assert self.client is not None
		# allow override from DB for phone
		phone: Optional[str] = None
		from .db import KVSetting
		async with get_session() as session:
			phone_db = await session.scalar(select(KVSetting.value).where(KVSetting.key == "telegram_phone"))
			if phone_db:
				phone = phone_db
		if not phone:
			log_bus.push("[auth] TELEGRAM_PHONE not set. Please specify it on the Settings page.")
			raise RuntimeError("TELEGRAM_PHONE not configured")
		try:
			await self.client.send_code_request(phone)
			log_bus.push("[auth] Code sent. Go to /auth and enter the code from Telegram.")
			code = await self._wait_for_auth_value("code")
			try:
				await self.client.sign_in(phone=phone, code=code)
			except errors.SessionPasswordNeededError:
				log_bus.push("[auth] 2FA enabled. Please enter the password on the /auth page.")
				password = await self._wait_for_auth_value("password")
				await self.client.sign_in(password=password)
			log_bus.push("[auth] Authorization successful.")
		except errors.ApiIdInvalidError as e:
			log_bus.push("[auth] Invalid TELEGRAM_API_ID/TELEGRAM_API_HASH pair. Get correct values from my.telegram.org (API development tools), save them in Settings, and click Restart. If a session was previously created, delete the sessions/account.session file.")
			raise
		except Exception as e:
			log_bus.push(f"[auth] Authorization error: {type(e).__name__}: {e}")
			raise

	async def send_auth_code(self) -> None:
		assert self.client is not None
		# use phone from DB
		from .db import KVSetting
		phone: Optional[str] = None
		async with get_session() as session:
			phone_db = await session.scalar(select(KVSetting.value).where(KVSetting.key == "telegram_phone"))
			if phone_db:
				phone = phone_db
		if not phone:
			log_bus.push("[auth] TELEGRAM_PHONE not set. Please specify it on the Settings page.")
			return
		await self.client.send_code_request(phone)
		log_bus.push("[auth] Code resent.")

	async def _wait_for_auth_value(self, kind: str) -> str:
		self._awaiting = kind
		self._auth_future = self.loop.create_future()
		try:
			return await self._auth_future
		finally:
			self._auth_future = None
			self._awaiting = None

	def provide_auth(self, value: str) -> None:
		fut = self._auth_future
		if fut and not fut.done():
			fut.set_result(value)

	async def _load_subscriptions(self) -> None:
		if not self.client or not self.client.is_connected():
			log_bus.push("[subs] Subscriptions update skipped: bot is not started. Will apply on Start/Restart.")
			return
		log_bus.push("[subs] LOAD_SUBS: begin")
		# удаляем ранее добавленные обработчики для нашего колбэка
		try:
			self.client.remove_event_handler(self._event_handler)
		except Exception:
			pass
		async with get_session() as session:
			res = await session.execute(select(Channel).where(Channel.enabled == 1))
			channels = res.scalars().all()
			for ch in channels:
				self.client.add_event_handler(self._event_handler, events.NewMessage(chats=ch.url))
		log_bus.push(f"[subs] Subscribed to {len(channels)} channels.")
		await self._cache_discussion_groups()
		log_bus.push("[subs] LOAD_SUBS: end")

	async def _cache_discussion_groups(self) -> None:
		assert self.client
		me = await self.client.get_me()
		log_bus.push(f"[cache] Start as {getattr(me, 'phone', 'unknown')}")
		# читаем список каналов из БД разом, чтобы не держать сессию во время сетевых вызовов
		async with get_session() as session:
			res = await session.execute(select(Channel).where(Channel.enabled == 1))
			channels = res.scalars().all()
		for idx, ch in enumerate(channels):
			# задержка только между каналами
			if idx > 0:
				try:
					min_d, max_d = await self._get_group_init_delays()
					delay = random.randint(min(min_d, max_d), max(min_d, max_d))
					if delay > 0:
						log_bus.push(f"[cache] Inter-channel delay {delay}s (range {min_d}..{max_d})")
						await asyncio.sleep(delay)
				except Exception:
					pass
			try:
				log_bus.push(f"  [{idx+1}/{len(channels)}] [cache] init {ch.url}")
				entity = await self.client.get_entity(ch.url)
				if hasattr(entity, 'broadcast') and entity.broadcast:
					# без задержек: сначала гарантируем вступление в сам канал
					try:
						in_channel = False
						try:
							_ = await self.client(GetParticipantRequest(channel=entity, participant='me'))
							in_channel = True
						except Exception:
							in_channel = False
						if in_channel:
							log_bus.push(f"    [cache] already in channel {getattr(entity, 'title', ch.url)} → skip join")
						else:
							if await self._should_auto_join_channels():
								await self.client(JoinChannelRequest(entity))
								log_bus.push(f"    [cache] joined channel {getattr(entity, 'title', ch.url)}")
							else:
								log_bus.push(f"    [cache] auto-join channels disabled → stay out of {getattr(entity, 'title', ch.url)}")
						# cache channel title regardless
						try:
							await self._remember_channel_name(entity, ch)
						except Exception:
							pass
					except errors.UserAlreadyParticipantError:
						pass
					except errors.FloodWaitError as e:
						log_bus.push(f"    [cache] Channel Join FloodWait {e.seconds}s → sleeping")
						await asyncio.sleep(e.seconds)
					except Exception as e:
						log_bus.push(f"    [cache] Channel Join error: {type(e).__name__}: {e}")
					full_entity = await self.client(GetFullChannelRequest(channel=entity))
					# remember human title if available
					try:
						await self._remember_channel_name(entity, ch)
					except Exception:
						pass
					if full_entity.full_chat.linked_chat_id:
						discussion_id = full_entity.full_chat.linked_chat_id
						self.channel_to_discussion[entity.id] = discussion_id
						log_bus.push(f"  -> [cache] {entity.title}: discussion group cached")
						# авто‑вступление при необходимости
						if await self._should_auto_join_discussions():
							group_entity = None
							try:
								group_entity = next((c for c in getattr(full_entity, 'chats', []) if getattr(c, 'id', None) == discussion_id), None)
							except Exception:
								group_entity = None
							if group_entity is None:
								group_entity = await self.client.get_entity(PeerChannel(discussion_id))
							# уже состоим?
							already = False
							try:
								_ = await self.client(GetParticipantRequest(channel=group_entity, participant='me'))
								already = True
							except Exception:
								already = False
							if already:
								log_bus.push(f"    [cache] already in discussion for {entity.title} → skip join")
							else:
								limit = await self._get_join_limit_per_hour()
								now = datetime.utcnow()
								cut = now - timedelta(hours=1)
								while self._join_attempts and self._join_attempts[0] < cut:
									self._join_attempts.popleft()
								if limit > 0 and len(self._join_attempts) >= limit:
									log_bus.push(f"    [cache] join limit per hour reached → skip join")
								else:
									try:
										min_interval = await self._get_join_min_interval()
										max_interval = await self._get_join_max_interval()
										interval = random.randint(min(min_interval, max_interval), max(min_interval, max_interval))
										if interval and self._last_join_at is not None:
											elapsed = (datetime.utcnow() - self._last_join_at).total_seconds()
											wait_secs = int(interval - elapsed)
											if wait_secs > 0:
												log_bus.push(f"    [cache] join interval → waiting {wait_secs}s")
												await asyncio.sleep(wait_secs)
									except Exception:
										pass
								self._join_attempts.append(datetime.utcnow())
								await self.client(JoinChannelRequest(group_entity))
								self._last_join_at = datetime.utcnow()
								log_bus.push(f"    [cache] joined discussion group for {entity.title}")
					else:
						log_bus.push(f"  -> [cache] {entity.title}: no linked discussion group")
				else:
					log_bus.push(f"  -> [cache] {ch.url}: not a broadcast channel")
			except errors.FloodWaitError as e:
				log_bus.push(f"[cache] FloodWait {e.seconds}s → sleeping")
				await asyncio.sleep(e.seconds)
			except Exception as e:
				log_bus.push(f"[cache] discussion error for {ch.url}: {type(e).__name__}: {e}")
				await asyncio.sleep(1)
		log_bus.push(f"[cache] Discussion map size: {len(self.channel_to_discussion)}")

	async def _event_handler(self, event):
		if not self.running:
			return

		# for media albums, telegram sends multiple events with same grouped_id.
		# we only want to process the first one.
		if event.grouped_id and event.grouped_id in self._processed_grouped_ids:
			return
		if event.grouped_id:
			self._processed_grouped_ids.append(event.grouped_id)

		if self.cool_down_until and datetime.utcnow() < self.cool_down_until:
			return
		metrics.inc("processed_posts_total")
		await self._process_post(event)

	async def _process_post(self, event, text_override: Optional[str] = None, photo_event_override=None):
		channel_id = event.chat_id
		channel_title = getattr(event.chat, 'title', f"ID:{channel_id}")
		# assign sequential index for this post
		self._post_counter += 1
		seq_idx = self._post_counter

		# apply per-channel skip-range at the very beginning: any post is skipped
		min_skip, max_skip = await self._get_skip_range()
		remaining = self.skip_counters.get(channel_id, 0)
		if remaining > 0:
			self.skip_counters[channel_id] = remaining - 1
			log_bus.push(f"[{seq_idx}] [{channel_title}] [skip] skipping post (left {remaining - 1})")
			metrics.inc("skipped_total")
			return

		if (event.message.video or event.message.video_note or event.message.voice or event.message.audio or
			event.message.document or event.message.poll or event.message.sticker or event.message.gif):
			log_bus.push(f"[{seq_idx}] [{channel_title}] [skip] unsupported content type")
			metrics.inc("skipped_total")
			return

		post_text = text_override if text_override is not None else (event.message.message or "")
		# optional: simulate reading the post
		try:
			if await self._is_simulate_read():
				await event.client.send_read_acknowledge(event.chat, max_id=event.message.id)
		except Exception:
			pass
		photo_to_process = photo_event_override or event
		image_base64: Optional[str] = None
		if photo_to_process.message.photo:
			try:
				img_bytes = await photo_to_process.message.download_media(file=bytes)
				if img_bytes:
					image_base64 = base64.b64encode(img_bytes).decode('utf-8')
			except Exception:
				image_base64 = None

		# concise start log (before any skips)
		preview = (post_text or "")[:25] + ("…" if (post_text and len(post_text) > 25) else "")
		log_bus.push(f"[{seq_idx}] [{channel_title}] [start] Post='{preview}', image={bool(image_base64)}")

		if fast_heuristic_is_ad(post_text):
			log_bus.push(f"[{seq_idx}] [{channel_title}] [skip] Heuristic: looks like ad → skip")
			metrics.inc("skipped_total")
			metrics.inc("skipped_heuristic_total")
			return

		# модель для классификации берём из ads_check_ai, а для генерации — из model
		ads_model = await self._get_ads_check_model()
		# общие sampling‑настройки (temperature/top_p/max_tokens) и prompt/model возьмём заранее
		prompt, gen_model, temp, top_p, max_tokens, gpt5_effort, gpt5_verbosity = await self._get_ai_settings()
		content_kind = (
			"image+text" if (image_base64 and post_text) else
			"image-only" if (image_base64 and not post_text) else
			"text-only" if (post_text and not image_base64) else
			"empty"
		)
		# отдельные настройки для проверки рекламы (перекрывают общие sampling)
		ads_prompt, ads_temp, ads_top_p, ads_max_tokens, ads_gpt5_effort, ads_gpt5_verbosity = await self._get_ads_ai_settings()
		ad_probability = await asyncio.to_thread(
			self.ai.classify_ad_probability,
			post_text,
			ads_model,
			image_base64,
			ads_temp if ads_temp is not None else temp,
			ads_top_p if ads_top_p is not None else top_p,
			ads_prompt,
			ads_max_tokens,
			ads_gpt5_effort,
			ads_gpt5_verbosity,
			channel_context=channel_title,
		) if self.ai else 0.0
		threshold = await self._get_threshold()
		if ad_probability > threshold:
			log_bus.push(f"[{seq_idx}] [{channel_title}] [skip] LLM says ad_prob={ad_probability:.2f} > {threshold:.2f} → skip")
			metrics.inc("skipped_total")
			metrics.inc("skipped_threshold_total")
			return

		with metrics.time_llm():
			comment = await asyncio.to_thread(
				self.ai.generate_comment,
				post_text=post_text,
				prompt_system=prompt,
				model=gen_model,
				temperature=temp,
				top_p=top_p,
				max_tokens=max_tokens,
				image_base64=image_base64,
				gpt5_reasoning_effort=gpt5_effort,
				gpt5_verbosity=gpt5_verbosity,
				channel_context=channel_title,
			)
			if not comment:
				log_bus.push(f"[{seq_idx}] [{channel_title}] [skip] OPENAI_API_KEY not configured → skip")
				metrics.inc("skipped_total")
				return

		# set next skip counter now (applies to both dry-run and real send)
		next_skip = random.randint(min(min_skip, max_skip), max(min_skip, max_skip))
		self.skip_counters[channel_id] = next_skip
		if await self._is_dry_run():
			log_bus.push(f"[{seq_idx}] [{channel_title}] [dry-run] would send → {comment[:120]} (next skip {next_skip})")
			metrics.inc("comments_sent_total")
			return

		# plan delays before sending (not applied in dry-run)
		min_d, max_d = await self._get_delays()
		delay = random.randint(min(min_d, max_d), max(min_d, max_d))
		# plan typing
		typing_secs = 0
		try:
			if await self._is_typing_enabled():
				min_t, max_t = await self._get_typing_delays()
				typing_secs = random.randint(min(min_t, max_t), max(min_t, max_t))
		except Exception:
			typing_secs = 0
		# single concise sending log
		log_bus.push(f"[{seq_idx}] [{channel_title}] [sending] Comment \"{comment}\" will be sent with sleep {delay}s and {typing_secs}s typing")
		if delay > 0:
			await asyncio.sleep(delay)

		with metrics.time_tg():
			await self._send_comment(event.client, event.chat_id, event.message.id, channel_title, comment, typing_secs=typing_secs, seq_idx=seq_idx)
		metrics.inc("comments_sent_total")

	async def _send_comment(self, client, channel_id: int, message_id: int, channel_name: str, comment_text: str, *, typing_secs: int = 0, seq_idx: int = 0):
		discussion_group_id = self.channel_to_discussion.get(channel_id)
		if not discussion_group_id:
			try:
				req = await client(GetDiscussionMessageRequest(peer=await client.get_input_entity(channel_id), msg_id=message_id))
				if req.chats:
					discussion_group_id = req.chats[0].id
					self.channel_to_discussion[channel_id] = discussion_group_id
			except Exception as e:
				log_bus.push(f"[{seq_idx}] [{channel_name}] [error] Cannot find discussion thread: {type(e).__name__}")
				metrics.inc("errors_total")
				return
		try:
			discussion_peer = PeerChannel(discussion_group_id)
			discussion_message = await client(GetDiscussionMessageRequest(peer=await client.get_input_entity(channel_id), msg_id=message_id))
			if discussion_message.messages:
				reply_to_msg_id = discussion_message.messages[0].id
				# typing simulation before sending (concise: no extra logs)
				try:
					if typing_secs > 0:
						async with client.action(discussion_peer, 'typing'):
							await asyncio.sleep(typing_secs)
				except Exception:
					pass
				await client.send_message(entity=discussion_peer, message=comment_text, reply_to=reply_to_msg_id)
				first = (comment_text or "")[:20]
				trail = "…" if comment_text and len(comment_text) > 20 else ""
				log_bus.push(f"[{seq_idx}] [{channel_name}] [success] Comment \"{first}{trail}\" sent")
			else:
				log_bus.push(f"[{seq_idx}] [{channel_name}] [send] No discussion message to reply")
		except errors.FloodWaitError as e:
			self.cool_down_until = datetime.utcnow() + timedelta(seconds=e.seconds)
			metrics.inc("floodwait_events_total")
			log_bus.push(f"[{seq_idx}] [{channel_name}] [send] FloodWait {e.seconds}s → cooldown")
		except errors.PeerFloodError:
			min_h, max_h = await self._get_peer_flood_cooldown_hours_range()
			hours = random.randint(min(min_h, max_h), max(min_h, max_h))
			self.cool_down_until = datetime.utcnow() + timedelta(hours=hours)
			log_bus.push(f"[{seq_idx}] [{channel_name}] [send] PeerFlood → long cooldown {hours}h")
		except Exception as e:
			metrics.inc("errors_total")
			log_bus.push(f"[{seq_idx}] [{channel_name}] [error] Send error: {type(e).__name__}: {e}")

	async def _get_ai_settings(self):
		from .db import KVSetting
		async with get_session() as session:
			prompt = await session.scalar(select(KVSetting.value).where(KVSetting.key == "prompt")) or "Ты — подписчик Telegram-канала. Оставь короткий комментарий."
			model = await session.scalar(select(KVSetting.value).where(KVSetting.key == "model")) or settings.model_default
			temp_v = await session.scalar(select(KVSetting.value).where(KVSetting.key == "temperature"))
			top_p_v = await session.scalar(select(KVSetting.value).where(KVSetting.key == "top_p"))
			max_tokens_v = await session.scalar(select(KVSetting.value).where(KVSetting.key == "max_tokens"))
			try:
				temp = float(temp_v) if temp_v is not None and temp_v != "" else None
			except Exception:
				temp = None
			try:
				top_p = float(top_p_v) if top_p_v is not None and top_p_v != "" else None
			except Exception:
				top_p = None
			try:
				max_tokens = int(max_tokens_v) if max_tokens_v is not None and max_tokens_v != "" else None
			except Exception:
				max_tokens = None
			gpt5_effort = await session.scalar(select(KVSetting.value).where(KVSetting.key == "gpt5_effort"))
			gpt5_verbosity = await session.scalar(select(KVSetting.value).where(KVSetting.key == "gpt5_verbosity"))
			return prompt, model, temp, top_p, max_tokens, gpt5_effort, gpt5_verbosity

	async def _get_ads_ai_settings(self):
		from .db import KVSetting
		async with get_session() as session:
			ads_prompt = await session.scalar(select(KVSetting.value).where(KVSetting.key == "ads_prompt"))
			ads_temp_v = await session.scalar(select(KVSetting.value).where(KVSetting.key == "ads_temperature"))
			ads_top_p_v = await session.scalar(select(KVSetting.value).where(KVSetting.key == "ads_top_p"))
			ads_max_tokens_v = await session.scalar(select(KVSetting.value).where(KVSetting.key == "ads_max_tokens"))
			ads_gpt5_effort = await session.scalar(select(KVSetting.value).where(KVSetting.key == "ads_gpt5_effort"))
			ads_gpt5_verbosity = await session.scalar(select(KVSetting.value).where(KVSetting.key == "ads_gpt5_verbosity"))
			try:
				ads_temp = float(ads_temp_v) if ads_temp_v is not None and ads_temp_v != "" else None
			except Exception:
				ads_temp = None
			try:
				ads_top_p = float(ads_top_p_v) if ads_top_p_v is not None and ads_top_p_v != "" else None
			except Exception:
				ads_top_p = None
			try:
				ads_max_tokens = int(ads_max_tokens_v) if ads_max_tokens_v is not None and ads_max_tokens_v != "" else None
			except Exception:
				ads_max_tokens = None
			return ads_prompt, ads_temp, ads_top_p, ads_max_tokens, ads_gpt5_effort, ads_gpt5_verbosity

	async def _remember_channel_name(self, entity, db_channel: Channel) -> None:
		try:
			name = getattr(entity, 'title', None) or getattr(entity, 'username', None) or ''
			name = str(name or '').strip()
			if not name:
				return
		except Exception:
			return
		from .db import get_session
		async with get_session() as session:
			obj = await session.get(Channel, db_channel.id)
			if obj and (not getattr(obj, 'channel_name', '') or obj.channel_name != name):
				obj.channel_name = name[:255]
				await session.commit()

	async def _get_ads_check_model(self) -> str:
		from .db import KVSetting
		async with get_session() as session:
			m = await session.scalar(select(KVSetting.value).where(KVSetting.key == "ads_check_ai"))
			if not m:
				m = await session.scalar(select(KVSetting.value).where(KVSetting.key == "model")) or settings.model_default
			return str(m)

	async def _get_delays(self):
		from .db import KVSetting
		async with get_session() as session:
			min_d = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "min_delay")) or 5))
			max_d = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "max_delay")) or 10))
			return min_d, max_d

	async def _get_group_init_delays(self) -> tuple[int, int]:
		from .db import KVSetting
		async with get_session() as session:
			min_d = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "group_init_min_delay")) or 1))
			max_d = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "group_init_max_delay")) or 3))
			return min_d, max_d

	async def _get_threshold(self) -> float:
		from .db import KVSetting
		async with get_session() as session:
			thr = float((await session.scalar(select(KVSetting.value).where(KVSetting.key == "ad_prob_thr")) or 0.7))
			return max(0.0, min(1.0, thr))

	async def _is_dry_run(self) -> bool:
		from .db import KVSetting
		async with get_session() as session:
			val = await session.scalar(select(KVSetting.value).where(KVSetting.key == "dry_run"))
			return str(val or "0") in ("1", "true", "True")

	async def _should_auto_join_discussions(self) -> bool:
		from .db import KVSetting
		async with get_session() as session:
			val = await session.scalar(select(KVSetting.value).where(KVSetting.key == "auto_join_discussions"))
			return str(val or "1") in ("1", "true", "True")

	async def _should_auto_join_channels(self) -> bool:
		from .db import KVSetting
		async with get_session() as session:
			val = await session.scalar(select(KVSetting.value).where(KVSetting.key == "auto_join_channels"))
			return str(val or "1") in ("1", "true", "True")

	async def _get_join_min_interval(self) -> int:
		from .db import KVSetting
		async with get_session() as session:
			val = await session.scalar(select(KVSetting.value).where(KVSetting.key == "join_min_interval_s"))
			try:
				return int(val or 600)
			except Exception:
				return 600

	async def _get_join_max_interval(self) -> int:
		from .db import KVSetting
		async with get_session() as session:
			val = await session.scalar(select(KVSetting.value).where(KVSetting.key == "join_max_interval_s"))
			try:
				return int(val or 1200)
			except Exception:
				return 1200

	async def _get_join_limit_per_hour(self) -> int:
		from .db import KVSetting
		async with get_session() as session:
			val = await session.scalar(select(KVSetting.value).where(KVSetting.key == "join_limit_per_hour"))
			try:
				limit = int(val or 10)
				return max(0, limit)
			except Exception:
				return 10

	async def _get_peer_flood_cooldown_hours_range(self) -> tuple[int, int]:
		from .db import KVSetting
		async with get_session() as session:
			min_h = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "peer_flood_cooldown_min_h")) or 12))
			max_h = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "peer_flood_cooldown_max_h")) or 48))
			return min_h, max_h

	async def _is_typing_enabled(self) -> bool:
		from .db import KVSetting
		async with get_session() as session:
			val = await session.scalar(select(KVSetting.value).where(KVSetting.key == "typing_enabled"))
			return str(val or "1") in ("1", "true", "True")

	async def _get_typing_delays(self) -> tuple[int, int]:
		from .db import KVSetting
		async with get_session() as session:
			min_t = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "typing_min")) or 1))
			max_t = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "typing_max")) or 5))
			return min_t, max_t

	async def _is_simulate_read(self) -> bool:
		from .db import KVSetting
		async with get_session() as session:
			val = await session.scalar(select(KVSetting.value).where(KVSetting.key == "simulate_read_enabled"))
			return str(val or "0") in ("1", "true", "True")

	async def _get_proxy_url(self) -> Optional[str]:
		from .db import KVSetting
		async with get_session() as session:
			return await session.scalar(select(KVSetting.value).where(KVSetting.key == "proxy_url"))

	async def _get_skip_range(self) -> tuple[int, int]:
		from .db import KVSetting
		async with get_session() as session:
			min_v = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "skip_min")) or 0))
			max_v = int((await session.scalar(select(KVSetting.value).where(KVSetting.key == "skip_max")) or 0))
			return min_v, max_v
