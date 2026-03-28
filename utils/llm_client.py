"""
OpenRouter LLM client (OpenAI-compatible chat completions).

Set in .env next to app.py:
  OPENROUTER_API_KEY=sk-or-v1-...
  OPENROUTER_MODEL=anthropic/claude-sonnet-4.5   # optional; see https://openrouter.ai/anthropic
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"


def _extract_openai_message_text(message: dict) -> str:
    """Normalize OpenAI/OpenRouter message content (string or multimodal list)."""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content).strip()


def _load_dotenv_if_present() -> None:
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if env_path.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except ImportError:
            pass


@runtime_checkable
class LLMClient(Protocol):
    """Interface for agent code (complete + JSON helpers)."""

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str: ...

    def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> dict: ...


class OpenRouterClient:
    """
    OpenRouter — https://openrouter.ai — chat completions API.
    Default model is Claude 3.5 Sonnet; override with OPENROUTER_MODEL.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
    ):
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        self.base_url = base_url.rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY", "").strip()
            or os.getenv("OPENROUTER_KEY", "").strip()
        )
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env (https://openrouter.ai/keys)."
            )
        logger.info(f"OpenRouter model: {self.model}")

    def _chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_APP_NAME", "Instacart BI Agent").strip()
        if title:
            headers["X-Title"] = title

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=300,
            )
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter request failed: {e}")

        if resp.status_code >= 400:
            err = data.get("error", data) if isinstance(data, dict) else data
            raise RuntimeError(f"OpenRouter API error ({resp.status_code}): {err}")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data}")

        msg = choices[0].get("message") or {}
        return _extract_openai_message_text(msg)

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        return self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> dict:
        system_with_json = (
            system_prompt
            + "\n\nCRITICAL: Respond ONLY with valid JSON. No markdown, no explanation, no ```json fences. Raw JSON only."
        )
        raw = self.complete(system_with_json, user_message, temperature=temperature)

        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```", "", cleaned).strip()

        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM. Raw output:\n{raw}")
            return {"error": "json_parse_failed", "raw": raw}


def get_llm_client() -> LLMClient:
    _load_dotenv_if_present()
    return OpenRouterClient()
