from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from archygazeisnotmortal.config import Settings
from archygazeisnotmortal.llm import LLMClient
from archygazeisnotmortal.media import (
    MAX_SHARE_ATTACHMENT_BYTES,
    detect_obvious_language,
    resolve_reply_media,
    sanitize_media_filename,
)
from archygazeisnotmortal.memory import (
    MemoryStore,
    SavedShareItem,
    clean_style_sample,
    format_messages_for_llm,
    is_owner_correction_feedback,
    make_share_slug,
)
from archygazeisnotmortal.persona import load_persona

logger = logging.getLogger(__name__)

DISCORD_MAX_MESSAGE_LEN = 2000


def split_discord_message(text: str, max_len: int = DISCORD_MAX_MESSAGE_LEN) -> list[str]:
    """Split text into chunks of at most `max_len` characters on clean boundaries."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_len:
        return [cleaned]

    chunks: list[str] = []
    remaining = cleaned
    while len(remaining) > max_len:
        window = remaining[:max_len]
        split_at = window.rfind("\n")
        if split_at < max_len // 2:
            split_at = window.rfind(" ")
        if split_at <= 0:
            split_at = max_len
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _is_dm_channel(channel: object) -> bool:
    if isinstance(channel, discord.DMChannel):
        return True
    return (
        isinstance(channel, discord.PartialMessageable)
        and channel.guild_id is None
        and channel.type in (None, discord.ChannelType.private)
    )


def normalize_discord_content(
    message: discord.Message, bot_user_id: int | None, bot_name: str
) -> str:
    content = message.content or ""
    if bot_user_id is not None:
        content = content.replace(f"<@{bot_user_id}>", f"@{bot_name}")
        content = content.replace(f"<@!{bot_user_id}>", f"@{bot_name}")

    if message.attachments:
        attachment_notes = ", ".join(att.filename for att in message.attachments[:5])
        suffix = f"[attached: {attachment_notes}]"
        content = f"{content}\n{suffix}".strip() if content.strip() else suffix

    stickers = getattr(message, "stickers", None) or ()
    if stickers:
        sticker_notes = ", ".join(s.name for s in stickers[:3])
        suffix = f"[sticker: {sticker_notes}]"
        content = f"{content}\n{suffix}".strip() if content.strip() else suffix

    return content.strip()


class PersonaBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents, help_command=None)

        self.settings = settings
        self.media_dir = (settings.database_path.parent / "media").resolve()
        self.memory = MemoryStore(
            db_path=settings.database_path,
            max_history_per_channel=settings.memory_window,
            ttl_hours=settings.memory_ttl_hours,
        )
        self.llm = LLMClient(settings)
        self.system_prompt = load_persona(
            settings.persona_file,
            persona_content=settings.persona_content,
        )

        self._channel_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._debounce_tasks: dict[int, asyncio.Task[None]] = {}
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._latest_trigger_message: dict[int, discord.Message] = {}
        self._last_reply_at: dict[int, float] = {}
        self._health_server: asyncio.Server | None = None

        self._register_slash_commands()

    async def _handle_health_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            try:
                await asyncio.wait_for(reader.read(1024), timeout=2.0)
            except TimeoutError:
                pass
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                b"Content-Length: 2\r\nConnection: close\r\n\r\nok"
            )
            await writer.drain()
        except OSError:
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def _start_health_server(self, port: int) -> None:
        self._health_server = await asyncio.start_server(
            self._handle_health_connection,
            host="0.0.0.0",
            port=port,
        )
        logger.info("Healthcheck HTTP server listening on port %d.", port)

    async def setup_hook(self) -> None:
        if self.settings.health_port is not None:
            await self._start_health_server(self.settings.health_port)
        await self.memory.init(self.settings.auto_reply_channel_ids)
        synced = await self.tree.sync()
        logger.info("Synced %d slash command(s).", len(synced))

    async def close(self) -> None:
        for task in list(self._active_tasks):
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._debounce_tasks.clear()
        self._active_tasks.clear()
        if self._health_server is not None:
            self._health_server.close()
            await self._health_server.wait_closed()
            self._health_server = None
        await self.llm.close()
        await self.memory.close()
        await super().close()

    async def on_ready(self) -> None:
        active_channels = await self.memory.list_allowed_channels()
        logger.info(
            "Logged in as %s (ID: %s) | Provider: %s/%s | Auto-reply channels: %s | DMs: %s",
            self.user,
            self.user.id if self.user else "?",
            self.settings.ai_provider,
            self.settings.ai_model,
            active_channels or "none yet (use /channel_add in Discord)",
            "enabled" if self.settings.reply_in_dms else "disabled",
        )
        if not self.settings.owner_user_ids:
            logger.warning(
                "OWNER_USER_IDS is empty; server members with Manage Server permission can run admin slash commands."
            )

    def _is_admin(self, interaction: discord.Interaction) -> bool:
        if self.settings.owner_user_ids:
            return interaction.user.id in self.settings.owner_user_ids
        if isinstance(interaction.user, discord.Member):
            return (
                interaction.user.guild_permissions.manage_guild
                or interaction.user.guild_permissions.administrator
            )
        return False

    def _build_turn_system_prompt(
        self,
        user_id: int,
        display_name: str,
        style_samples: Sequence[str] = (),
        learned_rules: Sequence[str] = (),
        saved_items: Sequence[SavedShareItem] = (),
        latest_text: str = "",
        is_dm: bool = False,
    ) -> str:
        clean_name = (display_name or "User").strip()
        if user_id in self.settings.special_user_ids:
            target_tag = f"{clean_name} (special_person)"
            target_role = "special_person"
            role_note = "This is the owner's special person."
        elif user_id in self.settings.owner_user_ids:
            target_tag = f"{clean_name} (owner_irl)"
            target_role = "owner_irl (the real human owner)"
            role_note = (
                "This is the real human owner talking in the chat (NOT special_person). "
                "Do NOT give them special_person or romantic treatment."
            )
        else:
            target_tag = clean_name
            target_role = "friend/member"
            role_note = "This is a regular friend/member (NOT special_person)."

        sections = [self.system_prompt]
        if style_samples:
            sample_lines = "\n".join(f"- {s}" for s in style_samples)
            sections.append(
                "--- LEARNED OWNER VOICE SAMPLES ---\n"
                "Real short Discord messages written by the owner. Mirror this exact texting rhythm, casing, slang, punctuation, and laugh words (do NOT treat unrelated facts here as current chat events):\n"
                f"{sample_lines}"
            )
        if learned_rules:
            rule_lines = "\n".join(f"- {r}" for r in learned_rules)
            sections.append(
                "--- OWNER CORRECTIONS & LEARNED RULES ---\n"
                "Explicit feedback and rules taught by the owner (follow these strictly, overriding conflicting habits above):\n"
                f"{rule_lines}"
            )
        if saved_items:
            item_lines: list[str] = []
            for item in saved_items:
                if item.file_path and item.content:
                    item_lines.append(
                        f"- \"{item.label}\": {item.content} (also attach saved file with [[send:{item.slug}]])"
                    )
                elif item.file_path:
                    item_lines.append(
                        f"- \"{item.label}\" (saved file) -> include [[send:{item.slug}]] to attach it"
                    )
                else:
                    item_lines.append(f"- \"{item.label}\" -> {item.content}")
            sections.append(
                "--- OWNER'S SAVED LINKS & MEDIA ---\n"
                "Real links, playlists, and media files saved by the owner to share when relevant:\n"
                + "\n".join(item_lines)
            )
        sections.append(
            "--- LINK & SONG SHARING RULES ---\n"
            "- NEVER invent, guess, or fake a URL, and NEVER stall with \"wait a sec\" / \"bentar ya\" promising a link without including it right now.\n"
            "- If a matching item exists in OWNER'S SAVED LINKS & MEDIA above, paste its exact URL verbatim or include `[[send:slug]]` to attach the saved file.\n"
            "- When someone asks for a song/playlist recommendation link and no saved playlist/link matches, write `[[song_link: Artist - Song Title]]` (max 3 per reply) — the bot replaces it automatically with a real playable YouTube/Apple Music link."
        )

        detected_lang = detect_obvious_language(latest_text) if latest_text else None
        lang_instruction = (
            f"CRITICAL LANGUAGE RULE: The latest message from `[{target_tag}]` is in {detected_lang}. "
            f"Reply STRICTLY in natural {detected_lang} (do NOT mix in words from another language)."
            if detected_lang
            else f"Match the exact language of `[{target_tag}]`'s latest message (e.g., if they just wrote in English, reply in English; if Indonesian, reply in Indonesian), even if earlier messages used a different language."
        )
        dm_note = (
            f"Chat context: Private 1-on-1 Discord DM with `[{target_tag}]` (nobody else is in this chat). Still remember not to share real IRL secrets or generate NSFW/explicit content.\n"
            if is_dm
            else ""
        )
        sections.append(
            "--- CURRENT REPLY TARGET ---\n"
            f"You are replying specifically to the latest message from `[{target_tag}]` "
            f"(user_id: {user_id}, role: {target_role}).\n"
            f"{dm_note}"
            f"{role_note}\n"
            f"{lang_instruction}\n"
            f"Keep speakers strictly separate: NEVER treat `[{target_tag}]` as another speaker "
            f"or attribute another speaker's messages to `[{target_tag}]`."
        )
        return "\n\n".join(sections)

    def _is_bot_mentioned_or_replied(self, message: discord.Message) -> bool:
        if self.user is None:
            return False
        if self.user in message.mentions:
            return True
        ref = message.reference
        if ref and isinstance(ref.resolved, discord.Message):
            return ref.resolved.author.id == self.user.id
        return False

    async def _should_process_message(self, message: discord.Message) -> bool:
        if message.author.bot:
            return False
        if message.type not in (discord.MessageType.default, discord.MessageType.reply):
            return False
        if self.settings.ignore_prefix and message.content.startswith(self.settings.ignore_prefix):
            return False

        is_dm = _is_dm_channel(message.channel)
        if is_dm:
            return self.settings.reply_in_dms

        if await self.memory.is_paused():
            return False

        if await self.memory.is_channel_allowed(message.channel.id):
            return True

        if self.settings.reply_on_mention and self._is_bot_mentioned_or_replied(message):
            return True

        return False

    async def _maybe_learn_from_owner(self, message: discord.Message) -> None:
        if not self.settings.learn_owner_style:
            return
        if message.author.bot:
            return
        if message.type not in (discord.MessageType.default, discord.MessageType.reply):
            return
        if message.author.id not in self.settings.owner_user_ids:
            return

        raw_content = (message.content or "").strip()
        if not raw_content:
            return
        if self.settings.ignore_prefix and raw_content.startswith(self.settings.ignore_prefix):
            return

        is_dm = _is_dm_channel(message.channel)
        replied_to_bot = is_dm or self._is_bot_mentioned_or_replied(message)
        in_allowed_channel = (
            False if is_dm else await self.memory.is_channel_allowed(message.channel.id)
        )
        if (in_allowed_channel or replied_to_bot) and is_owner_correction_feedback(
            raw_content, replied_to_bot=replied_to_bot
        ):
            clean_rule = " ".join(raw_content.split())
            if len(clean_rule) <= 220:
                await self.memory.add_learned_rule(clean_rule)
            return

        if is_dm:
            return

        sample = clean_style_sample(raw_content, ignore_prefix=self.settings.ignore_prefix)
        if sample:
            await self.memory.add_style_sample(sample, source_channel_id=message.channel.id)

    async def on_message(self, message: discord.Message) -> None:
        await self._maybe_learn_from_owner(message)

        if not await self._should_process_message(message):
            return

        bot_id = self.user.id if self.user else None
        bot_name = self.user.display_name if self.user else "Me"
        content = normalize_discord_content(message, bot_id, bot_name)
        if not content:
            return

        await self.memory.add_message(
            channel_id=message.channel.id,
            role="user",
            content=content,
            author_id=message.author.id,
            author_name=message.author.display_name,
        )

        channel_id = message.channel.id
        self._latest_trigger_message[channel_id] = message

        prev_debounce = self._debounce_tasks.get(channel_id)
        if prev_debounce and not prev_debounce.done():
            prev_debounce.cancel()

        task = asyncio.create_task(self._debounced_respond(channel_id))
        self._debounce_tasks[channel_id] = task
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def _debounced_respond(self, channel_id: int) -> None:
        current_task = asyncio.current_task()
        try:
            if self.settings.debounce_seconds > 0:
                await asyncio.sleep(self.settings.debounce_seconds)

            # Detach from _debounce_tasks before acquiring the lock so new incoming messages
            # queue a fresh debounce task instead of cancelling an in-flight API call.
            if self._debounce_tasks.get(channel_id) is current_task:
                self._debounce_tasks.pop(channel_id, None)

            trigger_msg = self._latest_trigger_message.get(channel_id)
            if trigger_msg is None:
                return

            async with self._channel_locks[channel_id]:
                is_dm = _is_dm_channel(trigger_msg.channel)
                if not is_dm and await self.memory.is_paused():
                    return

                elapsed = time.monotonic() - self._last_reply_at.get(channel_id, 0.0)
                remaining_cooldown = self.settings.channel_cooldown_seconds - elapsed
                if remaining_cooldown > 0:
                    await asyncio.sleep(remaining_cooldown)

                trigger_msg = self._latest_trigger_message.get(channel_id) or trigger_msg
                is_dm = _is_dm_channel(trigger_msg.channel)
                recent = await self.memory.get_recent_messages(
                    channel_id=channel_id,
                    limit=self.settings.memory_window,
                )
                formatted = format_messages_for_llm(
                    recent,
                    special_user_ids=self.settings.special_user_ids,
                    owner_user_ids=self.settings.owner_user_ids,
                )
                if not formatted or formatted[-1]["role"] != "user":
                    return

                bot_names = (
                    (self.user.display_name, self.user.name) if self.user else ()
                )
                style_samples = await self.memory.get_style_samples(limit=20)
                learned_rules = await self.memory.get_learned_rules(limit=20)
                saved_items = await self.memory.list_share_items(limit=40)
                turn_system_prompt = self._build_turn_system_prompt(
                    user_id=trigger_msg.author.id,
                    display_name=trigger_msg.author.display_name,
                    style_samples=style_samples,
                    learned_rules=learned_rules,
                    saved_items=saved_items,
                    latest_text=trigger_msg.content or "",
                    is_dm=is_dm,
                )

                async with trigger_msg.channel.typing():
                    raw_reply_text = await self.llm.generate_reply(
                        system_prompt=turn_system_prompt,
                        messages=formatted,
                        bot_names=bot_names,
                    )
                    reply_text, attachment_paths = await resolve_reply_media(
                        raw_reply_text,
                        saved_items=saved_items,
                        media_dir=self.media_dir,
                    )

                if not reply_text and not attachment_paths:
                    return

                stored_text = reply_text
                if attachment_paths:
                    attach_note = f"[attached: {', '.join(p.name for p in attachment_paths)}]"
                    stored_text = f"{reply_text}\n{attach_note}".strip() if reply_text else attach_note

                await self.memory.add_message(
                    channel_id=channel_id,
                    role="assistant",
                    content=stored_text,
                    author_id=self.user.id if self.user else None,
                    author_name=self.user.display_name if self.user else "Bot",
                )
                self._last_reply_at[channel_id] = time.monotonic()

                chunks = split_discord_message(reply_text)
                if not chunks and attachment_paths:
                    if not is_dm:
                        try:
                            await trigger_msg.reply(
                                files=[discord.File(p) for p in attachment_paths],
                                mention_author=False,
                            )
                            return
                        except discord.HTTPException:
                            pass
                    await trigger_msg.channel.send(
                        files=[discord.File(p) for p in attachment_paths]
                    )
                    return

                for i, chunk in enumerate(chunks):
                    files_kw = (
                        {"files": [discord.File(p) for p in attachment_paths]}
                        if i == 0 and attachment_paths
                        else {}
                    )
                    if i == 0 and not is_dm:
                        try:
                            await trigger_msg.reply(chunk, mention_author=False, **files_kw)
                            continue
                        except discord.HTTPException:
                            if files_kw:
                                files_kw = {"files": [discord.File(p) for p in attachment_paths]}
                    await trigger_msg.channel.send(chunk, **files_kw)

        except asyncio.CancelledError:
            if self._debounce_tasks.get(channel_id) is current_task:
                self._debounce_tasks.pop(channel_id, None)
            return
        except Exception:
            logger.exception("Failed to generate/send reply in channel %s", channel_id)

    def _register_slash_commands(self) -> None:
        @self.tree.command(name="channel_add", description="Enable auto-replies in a channel.")
        @app_commands.describe(channel="Channel to add (defaults to current channel)")
        async def channel_add(
            interaction: discord.Interaction,
            channel: discord.TextChannel | None = None,
        ) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            target = channel or interaction.channel
            if target is None:
                await interaction.response.send_message("No channel found.", ephemeral=True)
                return
            if _is_dm_channel(target):
                dm_state = "enabled" if self.settings.reply_in_dms else "disabled"
                await interaction.response.send_message(
                    f"DMs don't need `/channel_add` (DM replies are currently `{dm_state}` via `REPLY_IN_DMS`). Pick a server channel instead.",
                    ephemeral=True,
                )
                return
            added = await self.memory.add_allowed_channel(
                target.id, added_by=interaction.user.id
            )
            msg = (
                f"Enabled auto-replies in <#{target.id}>."
                if added
                else f"Auto-replies are already enabled in <#{target.id}>."
            )
            await interaction.response.send_message(msg, ephemeral=True)

        @self.tree.command(name="channel_remove", description="Disable auto-replies in a channel.")
        @app_commands.describe(channel="Channel to remove (defaults to current channel)")
        async def channel_remove(
            interaction: discord.Interaction,
            channel: discord.TextChannel | None = None,
        ) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            target = channel or interaction.channel
            if target is None:
                await interaction.response.send_message("No channel found.", ephemeral=True)
                return
            if _is_dm_channel(target):
                dm_state = "enabled" if self.settings.reply_in_dms else "disabled"
                await interaction.response.send_message(
                    f"DMs don't use `/channel_remove` (DM replies are currently `{dm_state}` via `REPLY_IN_DMS`). Pick a server channel instead.",
                    ephemeral=True,
                )
                return
            removed = await self.memory.remove_allowed_channel(target.id)
            msg = (
                f"Disabled auto-replies in <#{target.id}>."
                if removed
                else f"<#{target.id}> was not in the auto-reply list."
            )
            await interaction.response.send_message(msg, ephemeral=True)

        @self.tree.command(name="channel_list", description="List designated auto-reply channels.")
        async def channel_list(interaction: discord.Interaction) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            channels = await self.memory.list_allowed_channels()
            if not channels:
                await interaction.response.send_message(
                    "No auto-reply channels configured. Use `/channel_add` in a channel to add one.",
                    ephemeral=True,
                )
                return
            formatted = ", ".join(f"<#{cid}>" for cid in channels)
            await interaction.response.send_message(
                f"Auto-reply channels: {formatted}", ephemeral=True
            )

        @self.tree.command(
            name="persona_reload",
            description="Reload persona.md without restarting the bot.",
        )
        async def persona_reload(interaction: discord.Interaction) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            try:
                self.system_prompt = load_persona(
                    self.settings.persona_file,
                    persona_content=self.settings.persona_content,
                )
            except Exception as exc:
                await interaction.response.send_message(
                    f"Failed to reload persona: `{exc}`", ephemeral=True
                )
                return
            source = (
                self.settings.persona_file.name
                if self.settings.persona_file.exists()
                else "PERSONA_CONTENT"
            )
            await interaction.response.send_message(
                f"Reloaded persona from `{source}`.", ephemeral=True
            )

        @self.tree.command(
            name="memory_clear",
            description="Clear saved conversation memory in this channel or DM.",
        )
        async def memory_clear(interaction: discord.Interaction) -> None:
            is_dm = _is_dm_channel(interaction.channel)
            if not is_dm and not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            if interaction.channel_id is None:
                await interaction.response.send_message("No channel context found.", ephemeral=True)
                return
            deleted = await self.memory.clear_channel_memory(interaction.channel_id)
            await interaction.response.send_message(
                f"Cleared {deleted} stored message(s) for this conversation.", ephemeral=True
            )

        @self.tree.command(
            name="memory_clear_all",
            description="Wipe all saved conversation memory across every channel and DM.",
        )
        async def memory_clear_all(interaction: discord.Interaction) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            deleted = await self.memory.clear_all_memory()
            await interaction.response.send_message(
                f"Wiped {deleted} stored message(s) across all channels and DMs.",
                ephemeral=True,
            )

        @self.tree.command(
            name="bot_pause", description="Pause automatic replies in server channels."
        )
        async def bot_pause(interaction: discord.Interaction) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            await self.memory.set_paused(True)
            await interaction.response.send_message("Auto-replies paused.", ephemeral=True)

        @self.tree.command(
            name="bot_resume", description="Resume automatic replies in server channels."
        )
        async def bot_resume(interaction: discord.Interaction) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            await self.memory.set_paused(False)
            await interaction.response.send_message("Auto-replies resumed.", ephemeral=True)

        @self.tree.command(
            name="bot_status",
            description="Show AI provider, fallback, and channel status.",
        )
        async def bot_status(interaction: discord.Interaction) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            paused = await self.memory.is_paused()
            channels = await self.memory.list_allowed_channels()
            fallback = (
                f"{self.settings.fallback_ai_provider}/{self.settings.fallback_ai_model}"
                if self.settings.fallback_ai_provider
                else "none"
            )
            channel_text = ", ".join(f"<#{cid}>" for cid in channels) if channels else "none"
            ttl_text = (
                f"{self.settings.memory_ttl_hours:g}h auto-expiry"
                if self.settings.memory_ttl_hours > 0
                else "no auto-expiry"
            )
            samples_count, rules_count = await self.memory.count_style_memory()
            await interaction.response.send_message(
                (
                    f"**Status:** {'Paused (server channels)' if paused else 'Active'}\n"
                    f"**Direct Messages (DMs):** {'Enabled' if self.settings.reply_in_dms else 'Disabled'}\n"
                    f"**Primary AI:** `{self.settings.ai_provider}` / `{self.settings.ai_model}`\n"
                    f"**Fallback AI:** `{fallback}`\n"
                    f"**Auto-reply channels:** {channel_text}\n"
                    f"**Memory retention:** {self.settings.memory_window} msgs ({ttl_text})\n"
                    f"**Owner learning:** {samples_count} voice sample(s), {rules_count} rule(s) "
                    f"(auto-learn: {'on' if self.settings.learn_owner_style else 'off'})"
                ),
                ephemeral=True,
            )

        @self.tree.command(
            name="teach",
            description="Teach the bot a permanent rule or correction about how to talk.",
        )
        @app_commands.describe(rule="Rule or correction (e.g. 'never say wkwk, use lol instead')")
        async def teach(interaction: discord.Interaction, rule: str) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            cleaned = " ".join(rule.split())
            if not cleaned:
                await interaction.response.send_message("Rule cannot be empty.", ephemeral=True)
                return
            if len(cleaned) > 220:
                await interaction.response.send_message(
                    "Please keep each rule under 220 characters.", ephemeral=True
                )
                return
            await self.memory.add_learned_rule(cleaned)
            await interaction.response.send_message(f"Learned rule: `{cleaned}`", ephemeral=True)

        @self.tree.command(
            name="style_add",
            description="Save an example message showing how you naturally text.",
        )
        @app_commands.describe(example="Short example line in your real voice")
        async def style_add(interaction: discord.Interaction, example: str) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            cleaned = " ".join(example.split())
            if len(cleaned) < 2 or len(cleaned) > 180:
                await interaction.response.send_message(
                    "Please keep style examples between 2 and 180 characters.", ephemeral=True
                )
                return
            await self.memory.add_style_sample(cleaned, source_channel_id=interaction.channel_id)
            await interaction.response.send_message(
                f"Saved voice example: `{cleaned}`", ephemeral=True
            )

        @self.tree.command(
            name="style_sync",
            description="Scan recent messages sent by OWNER_USER_IDS in a channel to learn your style.",
        )
        @app_commands.describe(
            channel="Channel to scan (defaults to current channel)",
            limit="Recent channel messages to scan (10-200, default 100)",
        )
        async def style_sync(
            interaction: discord.Interaction,
            channel: discord.TextChannel | None = None,
            limit: app_commands.Range[int, 10, 200] = 100,
        ) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            if not self.settings.owner_user_ids:
                await interaction.response.send_message(
                    "Set `OWNER_USER_IDS` in `.env` so the bot knows whose messages to learn from.",
                    ephemeral=True,
                )
                return
            target = channel or interaction.channel
            if target is None or not hasattr(target, "history"):
                await interaction.response.send_message(
                    "Cannot read message history in this channel.", ephemeral=True
                )
                return
            await interaction.response.defer(thinking=True, ephemeral=True)
            try:
                scanned = 0
                candidates: list[str] = []
                async for msg in target.history(limit=limit):  # type: ignore[union-attr]
                    scanned += 1
                    if msg.author.bot or msg.author.id not in self.settings.owner_user_ids:
                        continue
                    sample = clean_style_sample(
                        msg.content, ignore_prefix=self.settings.ignore_prefix
                    )
                    if sample:
                        candidates.append(sample)
                added = 0
                for sample in reversed(candidates):
                    if await self.memory.add_style_sample(sample, source_channel_id=target.id):
                        added += 1
                total_samples, _ = await self.memory.count_style_memory()
                await interaction.followup.send(
                    f"Scanned {scanned} message(s) in <#{target.id}>; "
                    f"learned {added} new voice sample(s) ({total_samples} total saved).",
                    ephemeral=True,
                )
            except discord.Forbidden:
                await interaction.followup.send(
                    f"I don't have Read Message History permission in <#{target.id}>.",
                    ephemeral=True,
                )

        @self.tree.command(
            name="style_list",
            description="Show saved owner voice samples and learned rules.",
        )
        async def style_list(interaction: discord.Interaction) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            rules = await self.memory.get_learned_rules(limit=20)
            samples = await self.memory.get_style_samples(limit=20)
            if not rules and not samples:
                await interaction.response.send_message(
                    "No learned rules or voice samples saved yet. Try `/teach` or `/style_sync`.",
                    ephemeral=True,
                )
                return
            rules_block = "\n".join(f"- {r}" for r in rules) if rules else "_None yet_"
            samples_block = "\n".join(f"- {s}" for s in samples) if samples else "_None yet_"
            body = (
                f"**Learned Rules ({len(rules)}):**\n{rules_block}\n\n"
                f"**Voice Samples ({len(samples)} most recent):**\n{samples_block}"
            )
            if len(body) > 1950:
                body = body[:1947] + "..."
            await interaction.response.send_message(body, ephemeral=True)

        @self.tree.command(
            name="style_clear",
            description="Clear saved owner voice samples and/or learned rules.",
        )
        @app_commands.choices(
            target=[
                app_commands.Choice(name="All (samples + rules)", value="all"),
                app_commands.Choice(name="Voice samples only", value="samples"),
                app_commands.Choice(name="Learned rules only", value="rules"),
            ]
        )
        async def style_clear(
            interaction: discord.Interaction,
            target: str = "all",
        ) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            deleted_samples = (
                await self.memory.clear_style_samples() if target in ("all", "samples") else 0
            )
            deleted_rules = (
                await self.memory.clear_learned_rules() if target in ("all", "rules") else 0
            )
            await interaction.response.send_message(
                f"Cleared {deleted_samples} voice sample(s) and {deleted_rules} learned rule(s).",
                ephemeral=True,
            )

        @self.tree.command(
            name="share_add",
            description="Save a link, playlist, or file so the bot can share it in chat.",
        )
        @app_commands.describe(
            label="What this is (e.g. 'chill playlist', 'aphex twin pic', 'letterboxd')",
            link_or_text="URL or text snippet to share",
            file="Optional image/GIF/audio/video file (up to 8 MB) to attach when shared",
        )
        async def share_add(
            interaction: discord.Interaction,
            label: str,
            link_or_text: str | None = None,
            file: discord.Attachment | None = None,
        ) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            clean_label = " ".join(label.split())
            clean_content = (link_or_text or "").strip()
            if not clean_label:
                await interaction.response.send_message("Label cannot be empty.", ephemeral=True)
                return
            if not clean_content and file is None:
                await interaction.response.send_message(
                    "Provide a `link_or_text` URL/message or attach a `file`.", ephemeral=True
                )
                return

            saved_path_str = ""
            if file is not None:
                if file.size > MAX_SHARE_ATTACHMENT_BYTES:
                    await interaction.response.send_message(
                        "Attached file is too large (max 8 MB).", ephemeral=True
                    )
                    return
                await interaction.response.defer(thinking=True, ephemeral=True)
                slug = make_share_slug(clean_label)
                self.media_dir.mkdir(parents=True, exist_ok=True)
                dest = (self.media_dir / sanitize_media_filename(file.filename, slug)).resolve()
                if not dest.is_relative_to(self.media_dir):
                    await interaction.followup.send("Invalid filename.", ephemeral=True)
                    return
                await file.save(dest)
                saved_path_str = str(dest)
                item = await self.memory.add_share_item(
                    label=clean_label,
                    content=clean_content,
                    file_path=saved_path_str,
                    slug=slug,
                )
                await interaction.followup.send(
                    f"Saved `{item.label}` (id: `{item.slug}`) with attachment `{dest.name}`.",
                    ephemeral=True,
                )
                return

            item = await self.memory.add_share_item(label=clean_label, content=clean_content)
            await interaction.response.send_message(
                f"Saved `{item.label}` -> `{item.content}`", ephemeral=True
            )

        @self.tree.command(
            name="share_list",
            description="List saved links, playlists, and media files.",
        )
        async def share_list(interaction: discord.Interaction) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            items = await self.memory.list_share_items(limit=40)
            if not items:
                await interaction.response.send_message(
                    "No saved links or files yet. Use `/share_add` to add playlists, links, or pics.",
                    ephemeral=True,
                )
                return
            lines: list[str] = []
            for item in items:
                parts = []
                if item.content:
                    parts.append(item.content)
                if item.file_path:
                    parts.append(f"[file: {Path(item.file_path).name}]")
                lines.append(f"- **{item.label}** (`{item.slug}`): {' '.join(parts)}")
            body = "**Saved Links & Media:**\n" + "\n".join(lines)
            if len(body) > 1950:
                body = body[:1947] + "..."
            await interaction.response.send_message(body, ephemeral=True)

        @self.tree.command(
            name="share_remove",
            description="Delete a saved link/file by label or id (or 'all' to clear everything).",
        )
        @app_commands.describe(name="Saved label or slug to remove, or 'all'")
        async def share_remove(interaction: discord.Interaction, name: str) -> None:
            if not self._is_admin(interaction):
                await interaction.response.send_message(
                    "You are not authorized to run admin commands.", ephemeral=True
                )
                return
            if name.strip().lower() == "all":
                removed = await self.memory.clear_share_items()
                for item in removed:
                    if item.file_path:
                        p = Path(item.file_path).resolve()
                        if p.is_relative_to(self.media_dir) and p.is_file():
                            p.unlink(missing_ok=True)
                await interaction.response.send_message(
                    f"Removed {len(removed)} saved link/media item(s).", ephemeral=True
                )
                return

            removed_item = await self.memory.remove_share_item(name)
            if removed_item is None:
                await interaction.response.send_message(
                    f"No saved item matched `{name}`.", ephemeral=True
                )
                return
            if removed_item.file_path:
                p = Path(removed_item.file_path).resolve()
                if p.is_relative_to(self.media_dir) and p.is_file():
                    p.unlink(missing_ok=True)
            await interaction.response.send_message(
                f"Removed `{removed_item.label}`.", ephemeral=True
            )

        @self.tree.command(
            name="ask", description="Send a one-off message to the persona in any channel."
        )
        @app_commands.describe(message="What you want to say")
        async def ask(interaction: discord.Interaction, message: str) -> None:
            await interaction.response.defer(thinking=True)
            try:
                speaker = interaction.user.display_name
                if interaction.user.id in self.settings.special_user_ids:
                    speaker = f"{speaker} (special_person)"
                elif interaction.user.id in self.settings.owner_user_ids:
                    speaker = f"{speaker} (owner_irl)"
                prompt_msg = [
                    {
                        "role": "user",
                        "content": f"[{speaker}]: {message.strip()}",
                    }
                ]
                style_samples = await self.memory.get_style_samples(limit=20)
                learned_rules = await self.memory.get_learned_rules(limit=20)
                saved_items = await self.memory.list_share_items(limit=40)
                turn_system_prompt = self._build_turn_system_prompt(
                    user_id=interaction.user.id,
                    display_name=interaction.user.display_name,
                    style_samples=style_samples,
                    learned_rules=learned_rules,
                    saved_items=saved_items,
                    latest_text=message.strip(),
                    is_dm=_is_dm_channel(interaction.channel),
                )
                bot_names = (self.user.display_name, self.user.name) if self.user else ()
                raw_reply = await self.llm.generate_reply(
                    system_prompt=turn_system_prompt,
                    messages=prompt_msg,
                    bot_names=bot_names,
                )
                reply_text, attachment_paths = await resolve_reply_media(
                    raw_reply,
                    saved_items=saved_items,
                    media_dir=self.media_dir,
                )
                chunks = split_discord_message(reply_text)
                if not chunks and attachment_paths:
                    await interaction.followup.send(
                        files=[discord.File(p) for p in attachment_paths]
                    )
                    return
                if not chunks:
                    await interaction.followup.send("...")
                    return
                if attachment_paths:
                    await interaction.followup.send(
                        chunks[0], files=[discord.File(p) for p in attachment_paths]
                    )
                else:
                    await interaction.followup.send(chunks[0])
                for extra in chunks[1:]:
                    await interaction.followup.send(extra)
            except Exception as exc:
                logger.exception("/ask command failed")
                await interaction.followup.send(f"Couldn't generate a reply right now: `{exc}`")
