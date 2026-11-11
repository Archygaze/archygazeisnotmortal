from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

import discord
from discord import app_commands
from discord.ext import commands

from archygazeisnotmortal.config import Settings
from archygazeisnotmortal.llm import LLMClient
from archygazeisnotmortal.memory import MemoryStore, format_messages_for_llm
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

    return content.strip()


class PersonaBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents, help_command=None)

        self.settings = settings
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
            "Logged in as %s (ID: %s) | Provider: %s/%s | Auto-reply channels: %s",
            self.user,
            self.user.id if self.user else "?",
            self.settings.ai_provider,
            self.settings.ai_model,
            active_channels or "none yet (use /channel_add in Discord)",
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

    def _build_turn_system_prompt(self, user_id: int, display_name: str) -> str:
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
        return (
            f"{self.system_prompt}\n\n"
            "--- CURRENT REPLY TARGET ---\n"
            f"You are replying specifically to the latest message from `[{target_tag}]` "
            f"(user_id: {user_id}, role: {target_role}).\n"
            f"{role_note}\n"
            f"Keep speakers strictly separate: NEVER treat `[{target_tag}]` as another speaker "
            f"or attribute another speaker's messages to `[{target_tag}]`."
        )

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

        is_dm = isinstance(message.channel, discord.DMChannel)
        if is_dm:
            return self.settings.reply_in_dms

        if await self.memory.is_paused():
            return False

        if await self.memory.is_channel_allowed(message.channel.id):
            return True

        if self.settings.reply_on_mention and self._is_bot_mentioned_or_replied(message):
            return True

        return False

    async def on_message(self, message: discord.Message) -> None:
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
                is_dm = isinstance(trigger_msg.channel, discord.DMChannel)
                if not is_dm and await self.memory.is_paused():
                    return

                elapsed = time.monotonic() - self._last_reply_at.get(channel_id, 0.0)
                remaining_cooldown = self.settings.channel_cooldown_seconds - elapsed
                if remaining_cooldown > 0:
                    await asyncio.sleep(remaining_cooldown)

                trigger_msg = self._latest_trigger_message.get(channel_id) or trigger_msg
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
                turn_system_prompt = self._build_turn_system_prompt(
                    user_id=trigger_msg.author.id,
                    display_name=trigger_msg.author.display_name,
                )

                async with trigger_msg.channel.typing():
                    reply_text = await self.llm.generate_reply(
                        system_prompt=turn_system_prompt,
                        messages=formatted,
                        bot_names=bot_names,
                    )

                await self.memory.add_message(
                    channel_id=channel_id,
                    role="assistant",
                    content=reply_text,
                    author_id=self.user.id if self.user else None,
                    author_name=self.user.display_name if self.user else "Bot",
                )
                self._last_reply_at[channel_id] = time.monotonic()

                chunks = split_discord_message(reply_text)
                for i, chunk in enumerate(chunks):
                    if i == 0 and not is_dm:
                        try:
                            await trigger_msg.reply(chunk, mention_author=False)
                            continue
                        except discord.HTTPException:
                            pass
                    await trigger_msg.channel.send(chunk)

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
            is_dm = isinstance(interaction.channel, discord.DMChannel)
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
            await interaction.response.send_message(
                (
                    f"**Status:** {'Paused' if paused else 'Active'}\n"
                    f"**Primary AI:** `{self.settings.ai_provider}` / `{self.settings.ai_model}`\n"
                    f"**Fallback AI:** `{fallback}`\n"
                    f"**Auto-reply channels:** {channel_text}\n"
                    f"**Memory retention:** {self.settings.memory_window} msgs ({ttl_text})"
                ),
                ephemeral=True,
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
                turn_system_prompt = self._build_turn_system_prompt(
                    user_id=interaction.user.id,
                    display_name=interaction.user.display_name,
                )
                bot_names = (self.user.display_name, self.user.name) if self.user else ()
                reply_text = await self.llm.generate_reply(
                    system_prompt=turn_system_prompt,
                    messages=prompt_msg,
                    bot_names=bot_names,
                )
                chunks = split_discord_message(reply_text)
                if not chunks:
                    await interaction.followup.send("...")
                    return
                await interaction.followup.send(chunks[0])
                for extra in chunks[1:]:
                    await interaction.followup.send(extra)
            except Exception as exc:
                logger.exception("/ask command failed")
                await interaction.followup.send(f"Couldn't generate a reply right now: `{exc}`")
