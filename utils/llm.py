"""Thin wrapper around the OpenAI chat completions API.

All four agents go through `chat_json` so that model selection, retries,
temperature and JSON parsing live in exactly one place.
"""

from __future__ import annotations

import json
import logging
import os
import time

from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazily construct a single shared OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def chat_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> dict:
    """Call the model and return its response parsed as a JSON object.

    Uses OpenAI's JSON mode so the model is constrained to emit valid JSON.
    Retries on transient API errors and on malformed JSON.
    """
    client = get_client()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model or os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            return json.loads(content)
        except (OpenAIError, json.JSONDecodeError) as exc:
            last_error = exc
            wait = 2**attempt
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts") from last_error
