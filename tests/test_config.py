from pathlib import Path
import pytest

from archygazeisnotmortal.config import DEFAULT_MODELS, Settings


def test_settings_from_env_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "DISCORD_TOKEN",
        "AUTO_REPLY_CHANNEL_IDS",
        "OWNER_USER_IDS",
        "AI_PROVIDER",
        "AI_MODEL",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "FALLBACK_AI_PROVIDER",
        "FALLBACK_AI_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "DISCORD_TOKEN=tok123\n"
        "AUTO_REPLY_CHANNEL_IDS=111, 222, 111\n"
        "OWNER_USER_IDS=999\n"
        "AI_PROVIDER=gemini\n"
        "GEMINI_API_KEY=gem_abc\n"
        "FALLBACK_AI_PROVIDER=openrouter\n",
        encoding="utf-8",
    )

    settings = Settings.from_env(env_path=env_file, base_dir=tmp_path)
    assert settings.discord_token == "tok123"
    assert settings.auto_reply_channel_ids == (111, 222)
    assert settings.owner_user_ids == (999,)
    assert settings.ai_provider == "gemini"
    assert settings.ai_model == DEFAULT_MODELS["gemini"]
    assert settings.fallback_ai_provider == "openrouter"
    assert settings.fallback_ai_model == DEFAULT_MODELS["openrouter"]
    assert settings.persona_file == (tmp_path / "persona.md").resolve()
    assert settings.database_path == (tmp_path / "data" / "memory.db").resolve()


def test_settings_validation_catches_missing_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DISCORD_TOKEN", "AI_PROVIDER", "GROQ_API_KEY", "FALLBACK_AI_PROVIDER"):
        monkeypatch.delenv(var, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("AI_PROVIDER=groq\n", encoding="utf-8")

    settings = Settings.from_env(env_path=env_file, base_dir=tmp_path)
    errors = settings.validate_for_runtime()
    assert any("DISCORD_TOKEN" in e for e in errors)
    assert any("GROQ_API_KEY" in e for e in errors)


def test_settings_ollama_requires_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DISCORD_TOKEN", "AI_PROVIDER", "OLLAMA_BASE_URL", "FALLBACK_AI_PROVIDER"):
        monkeypatch.delenv(var, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("DISCORD_TOKEN=tok\nAI_PROVIDER=ollama\n", encoding="utf-8")

    settings = Settings.from_env(env_path=env_file, base_dir=tmp_path)
    assert settings.validate_for_runtime() == []
    assert settings.api_key_for("ollama") == "ollama"
    assert settings.base_url_for("ollama") == "http://localhost:11434/v1"


def test_settings_invalid_channel_id_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTO_REPLY_CHANNEL_IDS", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("AUTO_REPLY_CHANNEL_IDS=123,not_an_id\n", encoding="utf-8")

    with pytest.raises(ValueError, match="numeric Discord IDs"):
        Settings.from_env(env_path=env_file, base_dir=tmp_path)


def test_settings_privacy_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MEMORY_TTL_HOURS", "PERSONA_CONTENT", "PORT", "HEALTH_PORT"):
        monkeypatch.delenv(var, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "MEMORY_TTL_HOURS=12.5\nPERSONA_CONTENT=Name: archy\nPORT=8080\n",
        encoding="utf-8",
    )

    settings = Settings.from_env(env_path=env_file, base_dir=tmp_path)
    assert settings.memory_ttl_hours == 12.5
    assert settings.persona_content == "Name: archy"
    assert settings.health_port == 8080

    bad_env = tmp_path / ".env.bad"
    bad_env.write_text("MEMORY_TTL_HOURS=-1\n", encoding="utf-8")
    monkeypatch.delenv("MEMORY_TTL_HOURS", raising=False)
    with pytest.raises(ValueError, match="MEMORY_TTL_HOURS"):
        Settings.from_env(env_path=bad_env, base_dir=tmp_path)

