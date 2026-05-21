from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .config import ROOT_DIR


class LLMNotConfigured(RuntimeError):
    """Raised when an LLM call is requested without API configuration."""


@dataclass
class LLMResponse:
    content: str
    parsed: Optional[Dict[str, Any]]
    model: str
    cache_hit: bool
    elapsed_ms: int
    error: Optional[str] = None


class DeepSeekClient:
    """Small OpenAI-compatible client wrapper for DeepSeek Chat Completions."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        use_cache: bool = True,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        thinking: Optional[str] = None,
    ) -> None:
        _load_dotenv_if_available()
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.cache_dir = Path(cache_dir or ROOT_DIR / "outputs" / "cache")
        self.use_cache = use_cache
        self.timeout_seconds = float(timeout_seconds or os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60"))
        self.max_retries = int(max_retries or os.getenv("DEEPSEEK_MAX_RETRIES", "3"))
        self.reasoning_effort = reasoning_effort or os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
        self.thinking = thinking or os.getenv("DEEPSEEK_THINKING", "enabled")

    @property
    def available(self) -> bool:
        if not self.api_key or self.api_key == "your_deepseek_api_key":
            return False
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return True

    def json_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        cache_parts: Iterable[Any] = (),
        model: Optional[str] = None,
        task_name: str = "json",
    ) -> LLMResponse:
        if not self.available:
            raise LLMNotConfigured(
                "DeepSeek API is not configured. Set DEEPSEEK_API_KEY in .env or the environment."
            )

        active_model = model or self.model
        attempts: List[List[Dict[str, str]]] = [messages]
        attempts.append(
            messages
            + [
                {
                    "role": "user",
                    "content": "The previous response was not valid JSON. Return exactly one JSON object.",
                }
            ]
        )
        attempts.append(attempts[-1])

        last_response: Optional[LLMResponse] = None
        for index, attempt_messages in enumerate(attempts):
            thinking_enabled = index < 2
            response = self._chat_once(
                attempt_messages,
                cache_parts=[task_name, active_model, index, *cache_parts],
                model=active_model,
                thinking_enabled=thinking_enabled,
            )
            parsed = _extract_json_object(response.content)
            response.parsed = parsed
            if parsed is not None:
                return response
            last_response = response

        if last_response is None:
            raise RuntimeError("No LLM response was produced.")
        last_response.error = "json_parse_error"
        return last_response

    def _chat_once(
        self,
        messages: List[Dict[str, str]],
        *,
        cache_parts: Iterable[Any],
        model: str,
        thinking_enabled: bool,
    ) -> LLMResponse:
        cache_key = _cache_key([model, messages, *cache_parts])
        cache_path = self.cache_dir / f"{cache_key}.json"
        if self.use_cache and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return LLMResponse(
                content=payload["content"],
                parsed=None,
                model=payload.get("model", model),
                cache_hit=True,
                elapsed_ms=0,
            )

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_seconds)
        started = time.perf_counter()
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.max_retries)):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "stream": False,
                }
                if thinking_enabled:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                    kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
                completion = client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content or ""
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                response = LLMResponse(content=content, parsed=None, model=model, cache_hit=False, elapsed_ms=elapsed_ms)
                if self.use_cache:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(
                            {
                                "model": model,
                                "content": content,
                                "elapsed_ms": elapsed_ms,
                                "created_at": int(time.time()),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                return response
            except Exception as exc:  # pragma: no cover - depends on network/API.
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"LLM call failed: {last_error}") from last_error


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(ROOT_DIR / ".env")


def _cache_key(parts: Iterable[Any]) -> str:
    raw = json.dumps(list(parts), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(stripped[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None
