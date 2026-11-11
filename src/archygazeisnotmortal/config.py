from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SUPPORTED_PROVIDERS = ("groq", "gemini", "openrouter", "ollama", "openai_compatible")

DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-3.8-flash",
    "openrouter": "openrouter/free",
    "ollama": "llama3.2",
    "openai_compatible": "gpt-4o-mini",
}

DEFAULT_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def parse_id_set(value: str | None, field_name: str) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    ids: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(
                f"{field_name} must contain comma-separated numeric Discord IDs, got {part!r}"
            )
        val = int(part)
        if val not in ids:
            ids.append(val)
    return tuple(ids)


@dataclass(frozen=True, slots=True)
class Settings:
    discord_token: str
    auto_reply_channel_ids: tuple[int, ...]
    owner_user_ids: tuple[int, ...]
    reply_on_mention: bool
    reply_in_dms: bool
    ignore_prefix: str
    debounce_seconds: float
    channel_cooldown_seconds: float
    memory_window: int
    ai_provider: str
    ai_model: str
    groq_api_key: str
    gemini_api_key: str
    openrouter_api_key: str
    openai_api_key: str
    ollama_base_url: str
    openai_base_url: str
    fallback_ai_provider: str | None
    fallback_ai_model: str | None
    ai_temperature: float
    ai_max_tokens: int
    persona_file: Path
    database_path: Path
    memory_ttl_hours: float = 0.0
    persona_content: str = ""
    health_port: int | None = None
    special_user_ids: tuple[int, ...] = ()
    learn_owner_style: bool = True

    @classmethod
    def from_env(
        cls,
        env_path: str | Path | None = ".env",
        base_dir: Path | None = None,
    ) -> Settings:
        root = (base_dir or Path.cwd()).resolve()
        if env_path is not None:
            resolved_env = Path(env_path)
            if not resolved_env.is_absolute():
                resolved_env = root / resolved_env
            if resolved_env.exists():
                load_dotenv(resolved_env, override=False)

        provider = os.getenv("AI_PROVIDER", "groq").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported AI_PROVIDER {provider!r}. Expected one of: {', '.join(SUPPORTED_PROVIDERS)}"
            )

        model = os.getenv("AI_MODEL", "").strip() or DEFAULT_MODELS[provider]

        fallback_provider_raw = os.getenv("FALLBACK_AI_PROVIDER", "").strip().lower()
        fallback_provider: str | None = None
        fallback_model: str | None = None
        if fallback_provider_raw:
            if fallback_provider_raw not in SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Unsupported FALLBACK_AI_PROVIDER {fallback_provider_raw!r}. "
                    f"Expected one of: {', '.join(SUPPORTED_PROVIDERS)}"
                )
            fallback_provider = fallback_provider_raw
            fallback_model = (
                os.getenv("FALLBACK_AI_MODEL", "").strip()
                or DEFAULT_MODELS[fallback_provider]
            )

        persona_raw = Path(os.getenv("PERSONA_FILE", "persona.md").strip() or "persona.md")
        persona_file = persona_raw if persona_raw.is_absolute() else root / persona_raw

        db_raw = Path(os.getenv("DATABASE_PATH", "data/memory.db").strip() or "data/memory.db")
        database_path = db_raw if db_raw.is_absolute() else root / db_raw

        debounce_seconds = float(os.getenv("DEBOUNCE_SECONDS", "1.5").strip() or "1.5")
        cooldown_seconds = float(os.getenv("CHANNEL_COOLDOWN_SECONDS", "2.0").strip() or "2.0")
        memory_window = int(os.getenv("MEMORY_WINDOW", "24").strip() or "24")
        memory_ttl_hours = float(os.getenv("MEMORY_TTL_HOURS", "0").strip() or "0")
        ai_temperature = float(os.getenv("AI_TEMPERATURE", "0.8").strip() or "0.8")
        ai_max_tokens = int(os.getenv("AI_MAX_TOKENS", "350").strip() or "350")

        if debounce_seconds < 0:
            raise ValueError("DEBOUNCE_SECONDS must be >= 0")
        if cooldown_seconds < 0:
            raise ValueError("CHANNEL_COOLDOWN_SECONDS must be >= 0")
        if memory_window < 1:
            raise ValueError("MEMORY_WINDOW must be >= 1")
        if memory_ttl_hours < 0:
            raise ValueError("MEMORY_TTL_HOURS must be >= 0")
        if ai_max_tokens < 1:
            raise ValueError("AI_MAX_TOKENS must be >= 1")

        raw_port = (os.getenv("HEALTH_PORT") or os.getenv("PORT") or "").strip()
        health_port: int | None = None
        if raw_port:
            if not raw_port.isdigit() or not (1 <= int(raw_port) <= 65535):
                raise ValueError(f"PORT/HEALTH_PORT must be between 1 and 65535, got {raw_port!r}")
            health_port = int(raw_port)

        return cls(
            discord_token=os.getenv("DISCORD_TOKEN", "").strip(),
            auto_reply_channel_ids=parse_id_set(
                os.getenv("AUTO_REPLY_CHANNEL_IDS"), "AUTO_REPLY_CHANNEL_IDS"
            ),
            owner_user_ids=parse_id_set(
                os.getenv("OWNER_USER_IDS"), "OWNER_USER_IDS"
            ),
            reply_on_mention=parse_bool(os.getenv("REPLY_ON_MENTION"), True),
            reply_in_dms=parse_bool(os.getenv("REPLY_IN_DMS"), True),
            ignore_prefix=os.getenv("IGNORE_PREFIX", "//").strip(),
            debounce_seconds=debounce_seconds,
            channel_cooldown_seconds=cooldown_seconds,
            memory_window=memory_window,
            ai_provider=provider,
            ai_model=model,
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URLS["ollama"]).strip()
            or DEFAULT_BASE_URLS["ollama"],
            openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
            fallback_ai_provider=fallback_provider,
            fallback_ai_model=fallback_model,
            ai_temperature=ai_temperature,
            ai_max_tokens=ai_max_tokens,
            persona_file=persona_file,
            database_path=database_path,
            memory_ttl_hours=memory_ttl_hours,
            persona_content=os.getenv("PERSONA_CONTENT", "").strip(),
            health_port=health_port,
            special_user_ids=parse_id_set(
                os.getenv("SPECIAL_USER_IDS"), "SPECIAL_USER_IDS"
            ),
            learn_owner_style=parse_bool(os.getenv("LEARN_OWNER_STYLE"), True),
        )

    def api_key_for(self, provider: str) -> str:
        if provider == "groq":
            return self.groq_api_key
        if provider == "gemini":
            return self.gemini_api_key
        if provider == "openrouter":
            return self.openrouter_api_key
        if provider == "ollama":
            return self.openai_api_key or "ollama"
        if provider == "openai_compatible":
            return self.openai_api_key
        raise ValueError(f"Unknown provider: {provider}")

    def base_url_for(self, provider: str) -> str | None:
        if provider == "groq":
            return DEFAULT_BASE_URLS["groq"]
        if provider == "openrouter":
            return DEFAULT_BASE_URLS["openrouter"]
        if provider == "ollama":
            return self.ollama_base_url
        if provider == "openai_compatible":
            return self.openai_base_url or None
        if provider == "gemini":
            return None
        raise ValueError(f"Unknown provider: {provider}")

    def validate_for_runtime(self) -> list[str]:
        errors: list[str] = []
        if not self.discord_token:
            errors.append(
                "DISCORD_TOKEN is missing. Copy .env.example to .env and add your Discord bot token."
            )

        key_env_map = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai_compatible": "OPENAI_API_KEY",
        }
        if self.ai_provider in key_env_map and not self.api_key_for(self.ai_provider):
            errors.append(
                f"{key_env_map[self.ai_provider]} is required when AI_PROVIDER={self.ai_provider!r}."
            )

        if self.ai_provider == "openai_compatible" and not self.openai_base_url:
            errors.append(
                "OPENAI_BASE_URL is recommended/required when AI_PROVIDER='openai_compatible'."
            )

        if (
            self.fallback_ai_provider
            and self.fallback_ai_provider in key_env_map
            and not self.api_key_for(self.fallback_ai_provider)
        ):
            errors.append(
                f"{key_env_map[self.fallback_ai_provider]} is required when "
                f"FALLBACK_AI_PROVIDER={self.fallback_ai_provider!r}."
            )

        return errors

