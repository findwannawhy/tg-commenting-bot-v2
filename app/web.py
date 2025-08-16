from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Depends, Request, Response, status, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler as default_http_exception_handler
from fastapi.templating import Jinja2Templates

from .config import settings
from .db import init_db, get_session, Channel, KVSetting
from .logging_bus import log_bus
from .bot_runner import BotRunner
from .metrics import metrics
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError
from urllib.parse import quote


TEMPLATES_DIR = str(Path(__file__).parent / "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def auth_dep(request: Request):
	bearer = request.headers.get("Authorization", "")
	ok = False
	if bearer.startswith("Bearer ") and bearer.split(" ", 1)[1] == settings.admin_token:
		ok = True

	# Allow token in query param ONLY for log stream, as EventSource doesn't support headers
	if not ok and request.url.path == "/logs/stream":
		token_q = request.query_params.get("token")
		if token_q and token_q == settings.admin_token:
			ok = True

	if not ok:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
	return True


auth = Annotated[bool, Depends(auth_dep)]


class AppServer:
	def __init__(self) -> None:
		self.app = FastAPI()
		self.bot = BotRunner()
		self._setup_routes()

	def _setup_routes(self) -> None:
		app = self.app

		@app.exception_handler(HTTPException)
		async def _http_exc_handler(request: Request, exc: HTTPException):
			if exc.status_code == status.HTTP_401_UNAUTHORIZED:
				accept = request.headers.get("accept", "")
				if request.method == "GET" and ("text/html" in accept or "*/*" in accept):
					next_url = str(request.url)
					return RedirectResponse(url=f"/enter-token?next={quote(next_url, safe='')}", status_code=303)
				return JSONResponse({"detail": "Unauthorized"}, status_code=401)
			return await default_http_exception_handler(request, exc)

		@app.on_event("startup")
		async def _startup():
			await init_db()

		@app.get("/health")
		async def health():
			return {"status": "ok"}

		@app.get("/enter-token", response_class=HTMLResponse)
		async def enter_token(request: Request):
			next_url = request.query_params.get("next", "/")
			return templates.TemplateResponse("token.html", {"request": request, "next": next_url})

		@app.get("/", response_class=HTMLResponse)
		async def dashboard(request: Request, _: auth):
			return templates.TemplateResponse("dashboard.html", {"request": request})

		@app.get("/auth", response_class=HTMLResponse)
		async def auth_page(request: Request, _: auth):
			return templates.TemplateResponse("auth.html", {"request": request})

		@app.post("/auth")
		async def auth_submit(request: Request, _: auth, code: str = Form(None), password: str = Form(None), resend: str = Form(None)):
			if resend:
				await self.bot.send_auth_code()
			else:
				val = (code or password or "").strip()
				if val:
					self.bot.provide_auth(val)
			return RedirectResponse("/auth", status_code=303)

		@app.post("/auth/reset-session")
		async def auth_reset_session(request: Request, _: auth):
			await self.bot.reset_session()
			return RedirectResponse("/auth", status_code=303)

		@app.get("/metrics")
		async def metrics_endpoint(_: auth):
			return JSONResponse(metrics.snapshot())

		@app.post("/metrics/reset")
		async def metrics_reset(_: auth):
			metrics.reset()
			return JSONResponse({"status": "ok"})

		@app.get("/channels", response_class=HTMLResponse)
		async def channels_page(request: Request, _: auth):
			async with get_session() as session:
				res = await session.execute(select(Channel))
				items = res.scalars().all()
			return templates.TemplateResponse("channels.html", {"request": request, "channels": items})


		@app.post("/channels/add")
		async def channel_add(request: Request, _: auth, url: str = Form(...), note: str = Form("") ):
			u = url.strip()
			n = note.strip()
			async with get_session() as session:
				# если уже есть такой URL — просто обновим примечание
				res = await session.execute(select(Channel).where(Channel.url == u))
				existing = res.scalar_one_or_none()
				if existing:
					existing.note = n
					await session.commit()
				else:
					try:
						session.add(Channel(url=u, note=n))
						await session.commit()
					except IntegrityError:
						await session.rollback()
						# на случай гонки: если параллельно добавили — обновим note
						res = await session.execute(select(Channel).where(Channel.url == u))
						ex2 = res.scalar_one_or_none()
						if ex2:
							ex2.note = n
							await session.commit()
			await self.bot._load_subscriptions()
			return RedirectResponse("/channels", status_code=303)

		@app.post("/channels/{channel_id}/delete")
		async def channel_delete(channel_id: int, request: Request, _: auth):
			async with get_session() as session:
				await session.execute(delete(Channel).where(Channel.id == channel_id))
				await session.commit()
			await self.bot._load_subscriptions()
			return RedirectResponse("/channels", status_code=303)

		@app.post("/channels/{channel_id}/toggle")
		async def channel_toggle(channel_id: int, request: Request, _: auth):
			async with get_session() as session:
				ch = await session.get(Channel, channel_id)
				if ch:
					try:
						current = int(ch.enabled or 0)
					except Exception:
						current = 0
					ch.enabled = 0 if current == 1 else 1
					await session.commit()
			await self.bot._load_subscriptions()
			return RedirectResponse("/channels", status_code=303)

		@app.get("/settings", response_class=HTMLResponse)
		async def settings_page(request: Request, _: auth):
			async with get_session() as session:
				keys = [
					"prompt", "model", "ads_check_ai", "temperature", "top_p", "max_tokens",
					"gpt5_effort", "gpt5_verbosity",
					"ads_prompt", "ads_temperature", "ads_top_p", "ads_max_tokens",
					"ads_gpt5_effort", "ads_gpt5_verbosity",
					"min_delay", "max_delay", "ad_prob_thr", "dry_run", "proxy_url",
					"skip_min", "skip_max",
					"group_init_min_delay", "group_init_max_delay", "auto_join_discussions", "auto_join_channels",
					"typing_enabled", "typing_min", "typing_max", "simulate_read_enabled",
					"peer_flood_cooldown_min_h", "peer_flood_cooldown_max_h", "join_limit_per_hour", "join_min_interval_s", "join_max_interval_s",
					"telegram_api_id", "telegram_api_hash", "telegram_phone", "openai_api_key",
				]
				vals = {}
				for k in keys:
					vals[k] = await session.scalar(select(KVSetting.value).where(KVSetting.key == k))
			return templates.TemplateResponse("settings.html", {"request": request, "values": vals, "model_default": settings.model_default})

		@app.post("/settings")
		async def settings_save(request: Request, _: auth):
			form = await request.form()
			numeric_fields = {
				"temperature", "top_p", "max_tokens", "ads_temperature", "ads_top_p", "ads_max_tokens",
				"min_delay", "max_delay", "ad_prob_thr", "skip_min", "skip_max",
				"group_init_min_delay", "group_init_max_delay",
				"typing_min", "typing_max",
				"peer_flood_cooldown_min_h", "peer_flood_cooldown_max_h",
				"join_limit_per_hour", "join_min_interval_s", "join_max_interval_s",
			}
			async with get_session() as session:
				for k, v in form.items():
					obj = await session.get(KVSetting, k)
					
					final_v = str(v)
					if k in numeric_fields:
						if str(v).strip() == "":
							final_v = None
					
					if obj:
						obj.value = final_v
					else:
						session.add(KVSetting(key=k, value=final_v))
				await session.commit()
			log_bus.push("[web] Settings saved")
			return RedirectResponse("/settings", status_code=303)

		@app.get("/logs", response_class=HTMLResponse)
		async def logs_page(request: Request, _: auth):
			return templates.TemplateResponse("logs.html", {"request": request, "lines": log_bus.tail(300)})

		@app.get("/logs/stream")
		async def logs_stream(_: auth):
			async def gen():
				async for line in log_bus.stream():
					yield f"data: {line}\n\n"
			return StreamingResponse(gen(), media_type="text/event-stream")

		@app.post("/controls/start")
		async def ctl_start(request: Request, _: auth):
			try:
				await self.bot.start()
			except Exception as e:
				# не пробрасываем 500, логируем и возвращаемся на дашборд
				log_bus.push(f"[web] Start error: {type(e).__name__}: {e}")
			return RedirectResponse("/", status_code=303)

		@app.post("/controls/stop")
		async def ctl_stop(request: Request, _: auth):
			await self.bot.stop()
			return RedirectResponse("/", status_code=303)

		@app.post("/controls/restart")
		async def ctl_restart(request: Request, _: auth):
			await self.bot.restart()
			return RedirectResponse("/", status_code=303)


app_server = AppServer()
app = app_server.app
