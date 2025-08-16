from __future__ import annotations

import os
import time
import json
import re
from typing import Optional, List, Any, Dict, Tuple
from urllib.parse import urlparse
from openai import OpenAI

from .logging_bus import log_bus


class AIService:
    """
    Единый клиент для вызова OpenAI **Responses API** со всеми моделями.
    Логика параметров подстраивается под возможности конкретной модели:
      • Для reasoning‑моделей (GPT‑5, o‑series) — используем `reasoning.effort`,
        опционально `text.verbosity`. Параметры сэмплинга (`temperature`, `top_p`)
        не отправляем.
      • Для обычных моделей (gpt‑4o, gpt‑4.1, …) — отправляем `temperature`/`top_p`.
    """

    def __init__(self, api_key: Optional[str], default_model: str, proxy_url: Optional[str] = None) -> None:
        self._maybe_set_proxy_env(proxy_url)
        self.client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None
        self.default_model = default_model
        try:
            if self.client:
                log_bus.push(f"[ai] Initialized with model '{self.default_model}'.")
            else:
                log_bus.push(f"[ai] Initialized (no API key). Default model: '{self.default_model}'.")
        except Exception:
            pass

    # --- Сеть / прокси ------------------------------------------------------
    def _maybe_set_proxy_env(self, proxy_url: Optional[str]) -> None:
        if not proxy_url:
            return
        p = urlparse(proxy_url)
        if p.scheme in ("http", "https"):
            os.environ.setdefault("HTTPS_PROXY", proxy_url)
            os.environ.setdefault("HTTP_PROXY", proxy_url)
            try:
                log_bus.push(f"[ai] Proxy configured via {p.scheme.upper()} → {p.hostname}:{p.port}.")
            except Exception:
                pass

    # --- Классификация рекламы ----------------------------------------------
    def classify_ad_probability(
        self,
        text: str,
        model: Optional[str] = None,
        image_base64: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        gpt5_reasoning_effort: Optional[str] = None,
        gpt5_verbosity: Optional[str] = None,
        channel_context: Optional[str] = None,
    ) -> float:
        start_t = time.perf_counter()
        ch_prefix = f"[{channel_context}]" if channel_context else ""

        if not self.client:
            log_bus.push(f"{ch_prefix}[ads] OpenAI key not configured → returning 1.00 (client not created).")
            return 1.0

        prompt_text = (
            (prompt or "").strip()
            or "Проанализируй текст ниже и верни ОДНО число от 0.00 до 1.00, представляющее вероятность того, что это РЕКЛАМА."
        )
        log_bus.push(f"{ch_prefix}[ads] Start classification, post-preview='{self._shorten(text, 25)}', prompt-preview='{self._shorten(prompt_text, 25)}', image={bool(image_base64)}")
        model_to_use = model or self.default_model or "gpt-4o-mini"

        try:
            req_kwargs = self._build_responses_kwargs(
                model_name=model_to_use,
                instructions=prompt_text,
                user_parts=self._make_responses_user_parts(text=self._trim_for_ads(text, channel_context=channel_context), image_base64=image_base64),
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort=gpt5_reasoning_effort,
                verbosity=gpt5_verbosity,
                channel_context=channel_context,
            )

            out_text = self._responses_call("ads", req_kwargs, channel_context=channel_context)

            if not out_text:
                fb_model = "gpt-4o-mini"
                log_bus.push(f"{ch_prefix}[ads] Empty response → fallback to {fb_model} (Responses).")
                req_kwargs_fb = self._build_responses_kwargs(
                    model_name=fb_model,
                    instructions=prompt_text,
                    user_parts=self._make_responses_user_parts(text=self._trim_for_ads(text, channel_context=channel_context), image_base64=image_base64),
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    channel_context=channel_context,
                )
                out_text = self._responses_call("ads(fallback)", req_kwargs_fb, channel_context=channel_context)

            val = self._parse_probability(out_text or "", channel_context=channel_context)
            ad_p = max(0.0, min(1.0, val))
            dur = time.perf_counter() - start_t
            log_bus.push(f"{ch_prefix}[ads] Completed. Time={dur:.3f}s. Ad probability={ad_p:.3f}.")
            return ad_p
        except Exception as e:
            log_bus.push(f"{ch_prefix}[ads] Error {type(e).__name__}: {e}")
            dur = time.perf_counter() - start_t
            log_bus.push(f"{ch_prefix}[ads] Completed with error. Time={dur:.3f}s. Returning 0.00.")
            return 0.0

    # --- Генерация комментария ----------------------------------------------
    def generate_comment(
        self,
        post_text: str,
        prompt_system: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        image_base64: Optional[str] = None,
        gpt5_reasoning_effort: Optional[str] = None,
        gpt5_verbosity: Optional[str] = None,
        channel_context: Optional[str] = None,
    ) -> str:
        start_t = time.perf_counter()
        ch_prefix = f"[{channel_context}]" if channel_context else ""
        if not self.client:
            return ""

        model_to_use = model or self.default_model or "gpt-4o-mini"
        log_bus.push(f"{ch_prefix}[comment] Start generation, post-preview='{self._shorten(post_text, 25)}', prompt-preview='{self._shorten(prompt_system, 25)}', image={bool(image_base64)}")
        parts = self._make_responses_user_parts(text=(post_text or "")[:8000], image_base64=image_base64)
        req_kwargs = self._build_responses_kwargs(
            model_name=model_to_use,
            instructions=prompt_system or "",
            user_parts=parts,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=gpt5_reasoning_effort,
            verbosity=gpt5_verbosity,
            channel_context=channel_context,
        )
        text = self._responses_call("comment", req_kwargs, channel_context=channel_context)

        if not text:
            fb_model = "gpt-4o-mini"
            log_bus.push(f"{ch_prefix}[comment] Empty response → fallback to {fb_model} (Responses).")
            req_kwargs_fb = self._build_responses_kwargs(
                model_name=fb_model,
                instructions=prompt_system or "",
                user_parts=parts,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                channel_context=channel_context,
            )
            text = self._responses_call("comment(fallback)", req_kwargs_fb, channel_context=channel_context)

        normalized = self._normalize_comment_text(text or "", channel_context=channel_context)
        log_bus.push(f"{ch_prefix}[comment] Final comment text → '{normalized}'")
        dur = time.perf_counter() - start_t
        log_bus.push(
            f"{ch_prefix}[comment] Completed. Time={dur:.3f}s. Result length={len(normalized)}."
        )
        return normalized

    # --- Универсальный вызов Responses --------------------------------------
    def _build_responses_kwargs(
        self,
        *,
        model_name: str,
        instructions: str,
        user_parts: List[dict],
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        channel_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Собирает корректный для выбранной модели payload к Responses API."""
        caps = self._model_caps(model_name, channel_context=channel_context)
        ch_prefix = f"[{channel_context}]" if channel_context else ""

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "instructions": instructions,
            "input": [{"role": "user", "content": user_parts}],
        }

        if max_output_tokens is not None:
            mt = int(max_output_tokens)
            if caps["is_reasoning"] and mt < 16:
                log_bus.push(
                    f"{ch_prefix}[responses] Reasoning models require a reasonable minimum max_output_tokens; correcting {mt} → 16."
                )
                mt = 16
            kwargs["max_output_tokens"] = mt

        if caps["is_reasoning"]:
            # Игнорируем temperature/top_p — не поддерживаются reasoning‑семейством
            if temperature is not None or top_p is not None:
                log_bus.push(
                    f"{ch_prefix}[responses] Warning: temperature/top_p passed, but they are ignored for reasoning models."
                )
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            if verbosity:
                # GPT‑5: `text.verbosity`
                kwargs.setdefault("text", {})
                kwargs["text"]["verbosity"] = verbosity
        else:
            # Обычные модели: классический сэмплинг
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            if top_p is not None:
                kwargs["top_p"] = float(min(1.0, top_p))

        log_bus.push(f"{ch_prefix}[ai] Request model='{model_name}', reasoning={caps['is_reasoning']}, temperature={temperature}, top_p={top_p}, reasoning_effort={reasoning_effort}, verbosity={verbosity}")
        return kwargs

    def _responses_call(self, prefix: str, req_kwargs: Dict[str, Any], channel_context: Optional[str] = None) -> str:
        ch_prefix = f"[{channel_context}]" if channel_context else ""
        try:
            resp = self.client.responses.create(**req_kwargs)
            out_text = self._collect_responses_text(resp, channel_context=channel_context)
            self._log_usage_and_reqid(resp, prefix=prefix, channel_context=channel_context)
            log_bus.push(f"{ch_prefix}[{prefix}] Response received, text len={len(out_text)}")
            return out_text
        except Exception as e:
            log_bus.push(f"{ch_prefix}[{prefix}] Responses API error: {type(e).__name__}: {e}")
            return ""

    # --- Вспомогательные возможности моделей --------------------------------
    def _model_caps(self, model_name: str, channel_context: Optional[str] = None) -> Dict[str, Any]:
        name = (model_name or "").lower().strip()
        is_reasoning = (
            name.startswith("gpt-5") or  # GPT‑5 серия
            name.startswith("o1") or name.startswith("o3") or name.startswith("o4")  # o‑series, включая o4‑mini
        )
        return {
            "is_reasoning": is_reasoning,
        }

    # --- Разбор ответа Responses --------------------------------------------
    def _collect_responses_text(self, resp: Any, channel_context: Optional[str] = None) -> str:
        """Устойчивый сбор текста из Responses API."""
        # 1) Прямая попытка
        try:
            txt = (getattr(resp, "output_text", "") or "").strip()
            if txt:
                return txt
        except Exception:
            pass

        # 2) Разбор output[*].content[*].text
        try:
            out = getattr(resp, "output", None)
            if out and isinstance(out, (list, tuple)):
                buf: List[str] = []
                for item in out:
                    item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
                    if item_type and str(item_type) != "message":
                        continue
                    content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
                    if content and isinstance(content, (list, tuple)):
                        for part in content:
                            txt = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
                            if txt:
                                buf.append(str(txt))
                if buf:
                    return "".join(buf).strip()
        except Exception:
            pass

        # 3) Разбор через model_dump()
        try:
            d = resp.model_dump() if hasattr(resp, "model_dump") else None
            if isinstance(d, dict):
                candidates: List[str] = []

                def collect(obj: Any):
                    if obj is None:
                        return
                    if isinstance(obj, str):
                        s = obj.strip()
                        if s:
                            candidates.append(s)
                        return
                    if isinstance(obj, (list, tuple)):
                        for it in obj:
                            collect(it)
                        return
                    if isinstance(obj, dict):
                        for k in ("output_text", "text", "content", "value"):
                            if k in obj:
                                collect(obj[k])
                        return

                collect(d.get("output_text"))
                collect(d.get("output"))
                if not candidates:
                    collect(d)
                if candidates:
                    return "".join(candidates).strip()
        except Exception:
            pass

        return self._extract_responses_text(resp, channel_context=channel_context)

    def _log_usage_and_reqid(self, resp: Any, prefix: str = "RESP", channel_context: Optional[str] = None) -> None:
        ch_prefix = f"[{channel_context}]" if channel_context else ""
        try:
            rid = getattr(resp, "id", None) or getattr(resp, "_request_id", None)
            usage = getattr(resp, "usage", None)
            u_in = getattr(usage, "input_tokens", None)
            u_out = getattr(usage, "output_tokens", None)
            u_total = getattr(usage, "total_tokens", None)
            if u_in is not None or u_out is not None or rid:
                log_bus.push(f"{ch_prefix}[{prefix}] Meta — id={rid}, tokens: in={u_in}, out={u_out}, total={u_total}.")
        except Exception as e:
            log_bus.push(f"{ch_prefix}[{prefix}] Failed to log metadata: {e}")

    def _extract_responses_text(self, resp, channel_context: Optional[str] = None) -> str:
        candidates: List[str] = []

        def collect_strings(obj) -> None:
            if obj is None:
                return
            if isinstance(obj, str):
                s = obj.strip()
                if s:
                    candidates.append(s)
                return
            if isinstance(obj, (list, tuple)):
                for elem in obj:
                    collect_strings(elem)
                return
            if isinstance(obj, dict):
                if "value" in obj:
                    collect_strings(obj.get("value"))
                if "text" in obj:
                    collect_strings(obj.get("text"))
                if "content" in obj:
                    collect_strings(obj.get("content"))
                return
            # Объекты
            val = getattr(obj, "value", None)
            if val is not None:
                collect_strings(val)
            text = getattr(obj, "text", None)
            if text is not None:
                collect_strings(text)
            content = getattr(obj, "content", None)
            if content is not None:
                collect_strings(content)

        try:
            val = getattr(resp, "output_text", "")
            collect_strings(val)
        except Exception:
            pass

        try:
            out = getattr(resp, "output", None)
            if out and isinstance(out, (list, tuple)):
                for item in out:
                    content = getattr(item, "content", None)
                    if not content and isinstance(item, dict):
                        content = item.get("content")
                    if content:
                        for part in content:
                            collect_strings(getattr(part, "text", part.get("text") if isinstance(part, dict) else None))
        except Exception:
            pass

        try:
            content = getattr(resp, "content", None)
            if not content and isinstance(resp, dict):
                content = resp.get("content")
            if content:
                collect_strings(content)
        except Exception:
            pass

        if not candidates:
            return ""
        joined = "".join([s for s in candidates if isinstance(s, str)])
        return joined.strip()

    # --- Формирование входа -------------------------------------------------
    def _make_responses_user_parts(self, *, text: str, image_base64: Optional[str]) -> List[dict]:
        parts: List[dict] = []
        t = (text or "").strip()
        if t:
            parts.append({"type": "input_text", "text": t})
        if image_base64:
            parts.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_base64}"})
        return parts

    # --- Утилиты ------------------------------------------------------------
    def _parse_probability(self, content: str, channel_context: Optional[str] = None) -> float:
        s = (content or "").strip()
        ch_prefix = f"[{channel_context}]" if channel_context else ""
        try:
            val = float(s.replace(",", "."))
            return val
        except Exception:
            pass
        m = re.search(r"(?<!\d)(?:0(?:[\.,]\d{1,2})?|1(?:[\.,]00?)?)(?!\d)", s)
        if m:
            val = m.group(0).replace(",", ".")
            return float(val)
        log_bus.push(f"{ch_prefix}[ads] Failed to recognize probability in model response: '{self._shorten(s, 100)}'")
        raise ValueError("cannot parse probability")

    def _normalize_comment_text(self, raw: str, channel_context: Optional[str] = None) -> str:
        s = (raw or "").strip()
        if not s:
            return ""
        try:
            m = re.match(r"^```[a-zA-Z0-9_\-]*\n([\s\S]*?)\n```\s*$", s)
            if m:
                s = m.group(1).strip()
        except Exception:
            pass
        try:
            if s.startswith("{") or s.startswith("["):
                obj = json.loads(s)
                if isinstance(obj, dict):
                    cand = obj.get("comment") or obj.get("text") or obj.get("message")
                    if isinstance(cand, str):
                        s = cand.strip()
        except Exception:
            pass
        try:
            s = re.sub(r"^(?:Комментарий|Коммент|Ответ|Comment|Reply)\s*:\s*", "", s, flags=re.I).strip()
        except Exception:
            pass
        pairs = [("«", "»"), ("“", "”"), ('"', '"'), ("'", "'"), ("`", "`")]
        for left, right in pairs:
            if s.startswith(left) and s.endswith(right) and len(s) >= len(left) + len(right):
                s = s[len(left):-len(right)].strip()
        s = s.strip("` ")
        return s

    def _shorten(self, s: str, limit: int = 300) -> str:
        try:
            if not s:
                return ""
            ss = str(s)
            if len(ss) <= limit:
                return ss
            return ss[:limit] + "…"
        except Exception:
            return ""

    def _trim_for_ads(self, text: str, channel_context: Optional[str] = None) -> str:
        try:
            s = (text or "").strip()
            if len(s) <= 1000:
                return s
            return s[:700] + s[-300:]
        except Exception:
            return (text or "")[:1000]
