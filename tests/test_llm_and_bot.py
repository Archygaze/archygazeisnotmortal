from pathlib import Path
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from archygazeisnotmortal.bot import PersonaBot, normalize_discord_content, split_discord_message
from archygazeisnotmortal.config import Settings
from archygazeisnotmortal.llm import LLMClient, strip_speaker_prefix


def _make_settings(tmp_path: Path, **overrides: object) -> Settings:
    persona = tmp_path / "persona.md"
    persona.write_text("# Test Persona\n- Keep it short.", encoding="utf-8")
    defaults: dict[str, object] = {
        "discord_token": "fake_token",
        "auto_reply_channel_ids": (555,),
        "owner_user_ids": (123,),
        "reply_on_mention": True,
        "reply_in_dms": True,
        "ignore_prefix": "//",
        "debounce_seconds": 0.01,
        "channel_cooldown_seconds": 0.0,
        "memory_window": 10,
        "ai_provider": "groq",
        "ai_model": "llama-3.3-70b-versatile",
        "groq_api_key": "fake_groq",
        "gemini_api_key": "fake_gem",
        "openrouter_api_key": "",
        "openai_api_key": "",
        "ollama_base_url": "http://localhost:11434/v1",
        "openai_base_url": "",
        "fallback_ai_provider": "gemini",
        "fallback_ai_model": "gemini-3.8-flash",
        "ai_temperature": 0.7,
        "ai_max_tokens": 200,
        "persona_file": persona,
        "database_path": tmp_path / "memory.db",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_strip_speaker_prefix() -> None:
    assert strip_speaker_prefix("[Me]: hey there", ("Kai",)) == "hey there"
    assert strip_speaker_prefix("[Kai]: yeah for sure", ("Kai",)) == "yeah for sure"
    assert strip_speaker_prefix("[Other]: leave untouched", ("Kai",)) == "[Other]: leave untouched"
    assert strip_speaker_prefix("Note: keep colon", ("Kai",)) == "Note: keep colon"


def test_split_discord_message_respects_limit() -> None:
    text = ("word " * 500).strip()
    chunks = split_discord_message(text, max_len=2000)
    assert len(chunks) == 2
    assert all(len(c) <= 2000 for c in chunks)
    assert " ".join(chunks) == text


def test_normalize_discord_content_replaces_mention_and_attachments() -> None:
    msg = SimpleNamespace(
        content="<@999> look at this",
        attachments=[SimpleNamespace(filename="clip.png")],
    )
    normalized = normalize_discord_content(msg, bot_user_id=999, bot_name="Kai")  # type: ignore[arg-type]
    assert normalized == "@Kai look at this\n[attached: clip.png]"


async def test_llm_client_falls_back_on_primary_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path)
    client = LLMClient(settings)

    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("archygazeisnotmortal.llm.asyncio.sleep", fake_sleep)

    calls: list[str] = []

    async def fake_call_spec(spec, system_prompt, messages):
        calls.append(spec.provider)
        if spec.provider == "groq":
            raise RuntimeError("429 Rate Limit")
        return "[Bot]: all good, fallback worked"

    monkeypatch.setattr(client, "_call_spec", fake_call_spec)

    reply = await client.generate_reply(
        system_prompt="sys",
        messages=[{"role": "user", "content": "[Alex]: hey"}],
        bot_names=("Bot",),
    )
    assert calls == ["groq", "groq", "gemini"]
    assert reply == "all good, fallback worked"
    await client.close()


async def test_bot_auto_replies_and_coalesces_messages(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, debounce_seconds=0.05)
    bot = PersonaBot(settings)
    await bot.memory.init(settings.auto_reply_channel_ids)

    try:
        # Mock logged-in bot user and LLM response
        bot._connection.user = SimpleNamespace(id=999, name="MyClone", display_name="MyClone")  # type: ignore[attr-defined]
        bot.llm.generate_reply = AsyncMock(return_value="yeah im down in 10")  # type: ignore[method-assign]

        class DummyTyping:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        channel = MagicMock()
        channel.id = 555
        channel.typing = MagicMock(return_value=DummyTyping())
        channel.send = AsyncMock()

        author = SimpleNamespace(id=42, bot=False, display_name="Alex")
        msg1 = SimpleNamespace(
            author=author,
            channel=channel,
            type=discord.MessageType.default,
            content="yo",
            attachments=[],
            mentions=[],
            reference=None,
            reply=AsyncMock(),
        )
        msg2 = SimpleNamespace(
            author=author,
            channel=channel,
            type=discord.MessageType.default,
            content="u hopping on?",
            attachments=[],
            mentions=[],
            reference=None,
            reply=AsyncMock(),
        )
        ignored_msg = SimpleNamespace(
            author=author,
            channel=channel,
            type=discord.MessageType.default,
            content="// ignore this",
            attachments=[],
            mentions=[],
            reference=None,
            reply=AsyncMock(),
        )

        await bot.on_message(msg1)  # type: ignore[arg-type]
        await bot.on_message(msg2)  # type: ignore[arg-type]
        await bot.on_message(ignored_msg)  # type: ignore[arg-type]

        # Wait for debounce task to finish
        await bot._debounce_tasks[555]

        # Should coalesce msg1 + msg2 into ONE LLM call and reply to msg2
        assert bot.llm.generate_reply.await_count == 1
        call_kwargs = bot.llm.generate_reply.await_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "[Alex]: yo\n[Alex]: u hopping on?"}
        ]
        assert "CURRENT REPLY TARGET" in call_kwargs["system_prompt"]
        assert "`[Alex]` (user_id: 42, role: friend/member)" in call_kwargs["system_prompt"]
        assert "NOT special_person" in call_kwargs["system_prompt"]
        assert "Match the exact language" in call_kwargs["system_prompt"]
        msg1.reply.assert_not_awaited()
        msg2.reply.assert_awaited_once_with("yeah im down in 10", mention_author=False)
    finally:
        await bot.close()


async def test_bot_disambiguates_special_person_and_owner_irl(tmp_path: Path) -> None:
    settings = _make_settings(
        tmp_path,
        debounce_seconds=0.05,
        owner_user_ids=(123,),
        special_user_ids=(777,),
    )
    bot = PersonaBot(settings)
    await bot.memory.init(settings.auto_reply_channel_ids)

    try:
        bot._connection.user = SimpleNamespace(id=999, name="archygaze", display_name="archygaze")  # type: ignore[attr-defined]
        bot.llm.generate_reply = AsyncMock(return_value="bro stop narrating lol")  # type: ignore[method-assign]

        class DummyTyping:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        channel = MagicMock()
        channel.id = 555
        channel.typing = MagicMock(return_value=DummyTyping())
        channel.send = AsyncMock()

        special_author = SimpleNamespace(id=777, bot=False, display_name="belca")
        owner_author = SimpleNamespace(id=123, bot=False, display_name="archy")

        msg_special = SimpleNamespace(
            author=special_author,
            channel=channel,
            type=discord.MessageType.default,
            content="i love you",
            attachments=[],
            mentions=[],
            reference=None,
            reply=AsyncMock(),
        )
        msg_owner = SimpleNamespace(
            author=owner_author,
            channel=channel,
            type=discord.MessageType.default,
            content="he love you",
            attachments=[],
            mentions=[],
            reference=None,
            reply=AsyncMock(),
        )

        await bot.on_message(msg_special)  # type: ignore[arg-type]
        await bot.on_message(msg_owner)  # type: ignore[arg-type]
        await bot._debounce_tasks[555]

        assert bot.llm.generate_reply.await_count == 1
        call_kwargs = bot.llm.generate_reply.await_args.kwargs
        assert call_kwargs["messages"] == [
            {
                "role": "user",
                "content": "[belca (special_person)]: i love you\n[archy (owner_irl)]: he love you",
            }
        ]
        assert "CURRENT REPLY TARGET" in call_kwargs["system_prompt"]
        assert (
            "`[archy (owner_irl)]` (user_id: 123, role: owner_irl (the real human owner))"
            in call_kwargs["system_prompt"]
        )
        assert "NOT special_person" in call_kwargs["system_prompt"]
        msg_special.reply.assert_not_awaited()
        msg_owner.reply.assert_awaited_once_with("bro stop narrating lol", mention_author=False)
    finally:
        await bot.close()


async def test_bot_health_server_responds_200(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, health_port=0)
    bot = PersonaBot(settings)
    await bot._start_health_server(0)
    try:
        assert bot._health_server is not None
        port = bot._health_server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        response = await reader.read(1024)
        writer.close()
        await writer.wait_closed()
        assert b"200 OK" in response
        assert response.endswith(b"\r\n\r\nok")
    finally:
        await bot.close()


async def test_bot_learns_owner_style_and_feedback_rules(tmp_path: Path) -> None:
    settings = _make_settings(
        tmp_path,
        debounce_seconds=0.05,
        owner_user_ids=(123,),
        learn_owner_style=True,
    )
    bot = PersonaBot(settings)
    await bot.memory.init(settings.auto_reply_channel_ids)

    try:
        bot._connection.user = SimpleNamespace(id=999, name="Clone", display_name="Clone")  # type: ignore[attr-defined]
        bot.llm.generate_reply = AsyncMock(return_value="gotcha lol")  # type: ignore[method-assign]

        class DummyTyping:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        general_channel = MagicMock()
        general_channel.id = 888  # Not in auto_reply_channel_ids
        general_channel.typing = MagicMock(return_value=DummyTyping())
        general_channel.send = AsyncMock()

        bot_channel = MagicMock()
        bot_channel.id = 555  # In auto_reply_channel_ids
        bot_channel.typing = MagicMock(return_value=DummyTyping())
        bot_channel.send = AsyncMock()

        owner = SimpleNamespace(id=123, bot=False, display_name="Owner")
        friend = SimpleNamespace(id=456, bot=False, display_name="Friend")

        # Owner chatting in non-auto-reply channel teaches style sample without replying
        owner_general_msg = SimpleNamespace(
            author=owner,
            channel=general_channel,
            type=discord.MessageType.default,
            content="nah im listening to music rn",
            attachments=[],
            mentions=[],
            reference=None,
            reply=AsyncMock(),
        )
        # Friend chatting in non-auto-reply channel is NOT added to owner style samples
        friend_general_msg = SimpleNamespace(
            author=friend,
            channel=general_channel,
            type=discord.MessageType.default,
            content="friend phrase should not learn",
            attachments=[],
            mentions=[],
            reference=None,
            reply=AsyncMock(),
        )
        # Owner correcting bot in allowed channel is captured as a persistent learned rule
        owner_feedback_msg = SimpleNamespace(
            author=owner,
            channel=bot_channel,
            type=discord.MessageType.default,
            content="i told you im not using bad_laugh all the time",
            attachments=[],
            mentions=[],
            reference=None,
            reply=AsyncMock(),
        )

        await bot.on_message(owner_general_msg)  # type: ignore[arg-type]
        await bot.on_message(friend_general_msg)  # type: ignore[arg-type]
        await bot.on_message(owner_feedback_msg)  # type: ignore[arg-type]
        await bot._debounce_tasks[555]

        assert 888 not in bot._debounce_tasks
        assert await bot.memory.get_style_samples() == ["nah im listening to music rn"]
        assert await bot.memory.get_learned_rules() == [
            "i told you im not using bad_laugh all the time"
        ]

        # Later friend message in bot channel receives BOTH learned samples and rules in prompt
        friend_bot_msg = SimpleNamespace(
            author=friend,
            channel=bot_channel,
            type=discord.MessageType.default,
            content="wyd",
            attachments=[],
            mentions=[],
            reference=None,
            reply=AsyncMock(),
        )
        await bot.on_message(friend_bot_msg)  # type: ignore[arg-type]
        await bot._debounce_tasks[555]

        latest_system_prompt = bot.llm.generate_reply.await_args.kwargs["system_prompt"]
        assert "LEARNED OWNER VOICE SAMPLES" in latest_system_prompt
        assert "- nah im listening to music rn" in latest_system_prompt
        assert "OWNER CORRECTIONS & LEARNED RULES" in latest_system_prompt
        assert "- i told you im not using bad_laugh all the time" in latest_system_prompt
    finally:
        await bot.close()

