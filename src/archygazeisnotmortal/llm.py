from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from google import genai
from google.genai import types
from openai import AsyncOpenAI

from archygazeisnotmortal.config import Settings

logger = logging.getLogger(__name__)

SPEAKER_PREFIX_RE = re.compile(r"^(?:\[[^\]]{1,40}\]|Me|Assistant|Bot)\s*:\s*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    provider: str
    model: str
    api_key: str
    base_url: str | None


def strip_speaker_prefix(text: str, bot_names: tuple[str, ...] = ()) -> str:
    cleaned = text.strip()
    # Strip generic prefixes like "[BotName]: " or "Me: " if the model mimicked chat transcripts
    match = SPEAKER_PREFIX_RE.match(cleaned)
    if match:
        prefix_body = match.group(0).split(":", 1)[0].strip("[] ").lower()
        allowed_prefixes = {"me", "assistant", "bot", *(n.lower() for n in bot_names if n)}
        if prefix_body in allowed_prefixes:
            cleaned = cleaned[match.end() :].strip()
    return cleaned


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.primary = ProviderSpec(
            provider=settings.ai_provider,
            model=settings.ai_model,
            api_key=settings.api_key_for(settings.ai_provider),
            base_url=settings.base_url_for(settings.ai_provider),
        )
        self.fallback: ProviderSpec | None = None
        if settings.fallback_ai_provider and settings.fallback_ai_model:
            self.fallback = ProviderSpec(
                provider=settings.fallback_ai_provider,
                model=settings.fallback_ai_model,
                api_key=settings.api_key_for(settings.fallback_ai_provider),
                base_url=settings.base_url_for(settings.fallback_ai_provider),
            )
        self._openai_clients: dict[tuple[str, str | None], AsyncOpenAI] = {}
        self._gemini_clients: dict[str, genai.Client] = {}

    def _get_openai_client(self, spec: ProviderSpec) -> AsyncOpenAI:
        cache_key = (spec.provider, spec.base_url)
        client = self._openai_clients.get(cache_key)
        if client is None:
            headers = (
                {
                    "HTTP-Referer": "https://github.com/archygazeisnotmortal",
                    "X-Title": "archygazeisnotmortal",
                }
                if spec.provider == "openrouter"
                else None
            )
            client = AsyncOpenAI(
                api_key=spec.api_key,
                base_url=spec.base_url,
                timeout=30.0,
                default_headers=headers,
            )
            self._openai_clients[cache_key] = client
        return client

    def _get_gemini_client(self, spec: ProviderSpec) -> genai.Client:
        client = self._gemini_clients.get(spec.api_key)
        if client is None:
            client = genai.Client(
                api_key=spec.api_key,
                http_options=types.HttpOptions(timeout=30000),
            )
            self._gemini_clients[spec.api_key] = client
        return client

    async def close(self) -> None:
        for client in self._openai_clients.values():
            await client.close()
        self._openai_clients.clear()
        self._gemini_clients.clear()

    async def _call_spec(
        self,
        spec: ProviderSpec,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        if spec.provider == "gemini":
            client = self._get_gemini_client(spec)
            contents = [
                types.Content(
                    role="model" if m["role"] == "assistant" else "user",
                    parts=[types.Part.from_text(text=m["content"])],
                )
                for m in messages
            ]
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.settings.ai_temperature,
                max_output_tokens=self.settings.ai_max_tokens,
            )
            response = await client.aio.models.generate_content(
                model=spec.model,
                contents=contents,
                config=config,
            )
            return (response.text or "").strip()

        client = self._get_openai_client(spec)
        payload = [{"role": "system", "content": system_prompt}, *messages]
        response = await client.chat.completions.create(
            model=spec.model,
            messages=payload,  # type: ignore[arg-type]
            temperature=self.settings.ai_temperature,
            max_tokens=self.settings.ai_max_tokens,
        )
        if not response.choices:
            return ""
        return (response.choices[0].message.content or "").strip()

    async def generate_reply(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        bot_names: tuple[str, ...] = (),
    ) -> str:
        if not messages:
            raise ValueError("Cannot generate a reply from empty conversation history.")

        last_exc: Exception | None = None
        specs = [self.primary] + ([self.fallback] if self.fallback else [])

        for idx, spec in enumerate(specs):
            for attempt in range(2):
                try:
                    raw = await asyncio.wait_for(
                        self._call_spec(spec, system_prompt, messages),
                        timeout=35.0,
                    )
                    cleaned = strip_speaker_prefix(raw, bot_names=bot_names)
                    if cleaned:
                        return cleaned
                    raise RuntimeError(f"Empty response from {spec.provider}/{spec.model}")
                except Exception as exc:
                    last_exc = exc
                    is_last_attempt_for_spec = attempt == 1
                    logger.warning(
                        "LLM error on %s/%s (attempt %d): %s",
                        spec.provider,
                        spec.model,
                        attempt + 1,
                        exc,
                    )
                    if not is_last_attempt_for_spec:
                        await asyncio.sleep(1.0)
                    elif idx + 1 < len(specs):
                        logger.info(
                            "Switching to fallback provider %s/%s",
                            specs[idx + 1].provider,
                            specs[idx + 1].model,
                        )

        assert last_exc is not None
        raise last_exc

