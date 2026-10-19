"""OpenCode Go / OpenAI-compatible chat client with audit logging and cache."""

from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import api_key_source, get_api_key
from .io_utils import (
    append_jsonl,
    ensure_dir,
    read_json,
    sha256_text,
    utc_now_iso,
    write_json,
    write_jsonl,
)


@dataclass
class LLMResult:
    ok: bool
    text: str = ""
    error: str | None = None
    call_id: str | None = None
    cache_hit: bool = False
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    http_status: int | None = None


class LLMClient:
    def __init__(
        self,
        config: dict[str, Any],
        out_dir: str | Path,
        run_id: str,
        offline: bool = False,
        use_cache: bool | None = None,
    ) -> None:
        self.config = config
        self.generation = config.get("generation", {})
        self.out_dir = Path(out_dir)
        self.run_id = run_id
        self.offline = offline
        self.api_key = get_api_key()
        self.api_key_source = api_key_source()
        self.available = bool(self.api_key) and not offline
        self.required = bool(self.generation.get("llm_required", False))
        self.model = str(self.generation.get("model", "deepseek-v4.1-flash"))
        self.user_agent = str(self.generation.get("user_agent", "opencode-support-pipeline/1.0"))
        self.session_header = str(self.generation.get("session_header", "x-opencode-session"))
        self.session_id = str(
            self.generation.get("session_id") or re.sub(r"[^A-Za-z0-9._-]", "-", run_id)
        )
        self.base_url = str(self.generation.get("base_url", "https://opencode.ai/zen/go/v1")).rstrip("/")
        self.chat_path = str(self.generation.get("chat_path", "/chat/completions"))
        self.endpoint = self.base_url + ("" if self.chat_path.startswith("/") else "/") + self.chat_path
        self.use_cache = bool(self.generation.get("use_cache", True)) if use_cache is None else use_cache
        self.cache_dir = ensure_dir(
            self.out_dir / str(self.generation.get("cache_dir", "cache/llm_responses"))
        )
        self.log_path = self.out_dir / "llm_calls.jsonl"
        self.call_counter = 0

    def start_run(self) -> None:
        write_jsonl(self.log_path, [])

    def check_ready(self) -> None:
        if self.required and not self.available:
            reason = "offline mode was requested" if self.offline else "no API key was found"
            raise RuntimeError(
                "LLM generation is required (generation.llm_required=true), but "
                f"{reason}. Set OPENCODE_API_KEY / OPENCODE_GO_API_KEY or disable llm_required."
            )

    def mode(self) -> str:
        if self.offline:
            return "offline"
        if not self.api_key:
            return "offline_no_api_key"
        return "llm"

    def _log(self, record: dict[str, Any]) -> None:
        append_jsonl(self.log_path, record)

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            payload = read_json(path)
        except (ValueError, OSError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("response_text"), str):
            return None
        return payload

    def _write_cache(self, cache_key: str, prompt_hash: str, text: str, usage: dict[str, Any]) -> None:
        write_json(
            self._cache_path(cache_key),
            {
                "cache_key": cache_key,
                "model": self.model,
                "prompt_sha256": prompt_hash,
                "response_text": text,
                "usage": usage,
                "created_at": utc_now_iso(),
            },
        )

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return str(content or "")

    branch_retry_statuses = {408, 409, 425, 429, 500, 502, 503, 504}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stage: str,
        input_artifacts: list[str] | None = None,
        output_artifact: str | None = None,
    ) -> LLMResult:
        prompt_hash = sha256_text(system_prompt + "\n\n" + user_prompt)
        cache_key = sha256_text(self.model + "\n" + prompt_hash)
        self.call_counter += 1
        call_id = f"{self.run_id}-llm-{self.call_counter:03d}"

        if not self.available:
            return LLMResult(
                ok=False,
                error="llm_unavailable",
                call_id=call_id,
                model=self.model,
            )

        if self.use_cache:
            cached = self._read_cache(cache_key)
            if cached:
                self._log(
                    {
                        "call_id": call_id,
                        "run_id": self.run_id,
                        "stage": stage,
                        "timestamp": utc_now_iso(),
                        "provider": self.generation.get("provider", "opencode-go"),
                        "model": self.model,
                        "session_id": self.session_id,
                        "endpoint": self.endpoint,
                        "prompt_sha256": prompt_hash,
                        "prompt_chars": len(system_prompt) + len(user_prompt),
                        "input_artifacts": input_artifacts or [],
                        "output_artifact": output_artifact,
                        "temperature": self.generation.get("temperature", 0),
                        "max_tokens": self.generation.get("max_tokens"),
                        "latency_ms": 0,
                        "status": "cache_hit",
                        "cache_hit": True,
                        "http_status": None,
                        "usage": cached.get("usage", {}),
                        "response_sha256": sha256_text(cached["response_text"]),
                        "error": None,
                        "attempts": 0,
                    }
                )
                return LLMResult(
                    ok=True,
                    text=cached["response_text"],
                    call_id=call_id,
                    cache_hit=True,
                    model=self.model,
                    usage=cached.get("usage", {}),
                    latency_ms=0,
                )

        max_retries = int(self.generation.get("max_retries", 2))
        backoff = float(self.generation.get("retry_backoff_seconds", 1.0))
        timeout = float(self.generation.get("timeout_seconds", 60))
        temperature = float(self.generation.get("temperature", 0))
        max_tokens = int(self.generation.get("max_tokens", 500))
        use_response_format = bool(self.generation.get("response_format_json", True))

        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if use_response_format:
            request_payload["response_format"] = {"type": "json_object"}

        last_error: str | None = None
        last_status: int | None = None
        started = time.monotonic()
        attempts = 0

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            try:
                request = urllib.request.Request(
                    self.endpoint,
                    data=json.dumps(request_payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": self.user_agent,
                        self.session_header: self.session_id,
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                    status = int(getattr(response, "status", 200))
                payload = json.loads(raw)
                content = self._extract_content(payload)
                if not content.strip():
                    raise ValueError("model returned an empty message")
                usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
                latency_ms = int((time.monotonic() - started) * 1000)
                self._log(
                    {
                        "call_id": call_id,
                        "run_id": self.run_id,
                        "stage": stage,
                        "timestamp": utc_now_iso(),
                        "provider": self.generation.get("provider", "opencode-go"),
                        "model": self.model,
                        "session_id": self.session_id,
                        "endpoint": self.endpoint,
                        "prompt_sha256": prompt_hash,
                        "prompt_chars": len(system_prompt) + len(user_prompt),
                        "input_artifacts": input_artifacts or [],
                        "output_artifact": output_artifact,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "latency_ms": latency_ms,
                        "status": "success",
                        "cache_hit": False,
                        "http_status": status,
                        "usage": usage,
                        "response_sha256": sha256_text(content),
                        "error": None,
                        "attempts": attempts,
                    }
                )
                if self.use_cache:
                    self._write_cache(cache_key, prompt_hash, content, usage)
                return LLMResult(
                    ok=True,
                    text=content,
                    call_id=call_id,
                    cache_hit=False,
                    model=self.model,
                    usage=usage,
                    latency_ms=latency_ms,
                    http_status=status,
                )
            except urllib.error.HTTPError as exc:
                last_status = int(exc.code)
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:  # pragma: no cover - defensive
                    body = ""
                last_error = f"HTTP {exc.code}: {body[:500]}"
                if exc.code in (400, 422) and use_response_format:
                    use_response_format = False
                    request_payload.pop("response_format", None)
                    continue
                if exc.code in self.branch_retry_statuses and attempt < max_retries:
                    time.sleep(backoff * (2**attempt))
                    continue
                break
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                last_error = f"network error: {exc}"
                if attempt < max_retries:
                    time.sleep(backoff * (2**attempt))
                    continue
                break
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = f"response parsing error: {exc}"
                if attempt < max_retries:
                    time.sleep(backoff * (2**attempt))
                    continue
                break
            except Exception as exc:  # pragma: no cover - defensive
                last_error = f"unexpected error: {exc}"
                break

        latency_ms = int((time.monotonic() - started) * 1000)
        self._log(
            {
                "call_id": call_id,
                "run_id": self.run_id,
                "stage": stage,
                "timestamp": utc_now_iso(),
                "provider": self.generation.get("provider", "opencode-go"),
                "model": self.model,
                "session_id": self.session_id,
                "endpoint": self.endpoint,
                "prompt_sha256": prompt_hash,
                "prompt_chars": len(system_prompt) + len(user_prompt),
                "input_artifacts": input_artifacts or [],
                "output_artifact": output_artifact,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "latency_ms": latency_ms,
                "status": "error",
                "cache_hit": False,
                "http_status": last_status,
                "usage": {},
                "response_sha256": None,
                "error": last_error,
                "attempts": attempts,
            }
        )
        return LLMResult(
            ok=False,
            error=last_error or "unknown LLM error",
            call_id=call_id,
            model=self.model,
            latency_ms=latency_ms,
            http_status=last_status,
        )
