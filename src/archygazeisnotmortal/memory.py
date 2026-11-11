from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

_MENTION_OR_CHANNEL_RE = re.compile(r"<(?:@!?|@&|#)\d+>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:(\w+):\d+>")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

_FEEDBACK_PHRASES = (
    "i told you",
    "dont use ",
    "don't use ",
    "not using ",
    "dont say ",
    "don't say ",
    "stop saying ",
    "stop using ",
    "never say ",
    "never use ",
    "i dont use ",
    "i don't use ",
    "i dont say ",
    "i don't say ",
    "i never say ",
    "i never use ",
    "im not this ",
    "i'm not this ",
    "not how i talk",
    "dont talk like",
    "don't talk like",
    "jangan pake ",
    "jangan pakai ",
    "jangan bilang ",
    "gw ga pake ",
    "gw gak pake ",
    "gue ga pake ",
    "gue gak pake ",
    "aku ga pake ",
    "aku gak pake ",
    "ga pernah pake ",
    "gak pernah pake ",
)

_REPLY_ONLY_FEEDBACK_PHRASES = (
    "stop being ",
    "dont be so ",
    "don't be so ",
    "talk like me",
)


def is_owner_correction_feedback(text: str, replied_to_bot: bool = False) -> bool:
    """Detect explicit style/behavior feedback directed at the bot by its owner."""
    lowered = " ".join(text.lower().split())
    if not lowered or len(lowered) > 220:
        return False
    if any(phrase in lowered for phrase in _FEEDBACK_PHRASES):
        return True
    if replied_to_bot and any(phrase in lowered for phrase in _REPLY_ONLY_FEEDBACK_PHRASES):
        return True
    return False


def clean_style_sample(raw_content: str, ignore_prefix: str = "//") -> str | None:
    """Normalize a raw Discord message into a safe short texting-style sample."""
    text = (raw_content or "").strip()
    if not text:
        return None
    if ignore_prefix and text.startswith(ignore_prefix):
        return None
    if text.startswith(("/", "!")):
        return None

    text = _MENTION_OR_CHANNEL_RE.sub("", text)
    text = _CUSTOM_EMOJI_RE.sub(r":\1:", text)
    text = _URL_RE.sub("", text)
    text = " ".join(text.split())

    if len(text) < 4 or len(text) > 180:
        return None
    if is_owner_correction_feedback(text, replied_to_bot=False):
        return None
    return text


@dataclass(frozen=True, slots=True)
class ChatMessage:
    id: int
    channel_id: int
    role: str
    author_id: int | None
    author_name: str | None
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class SavedShareItem:
    id: int
    slug: str
    label: str
    content: str
    file_path: str
    created_at: str


def make_share_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug[:60] or "item"


def format_messages_for_llm(
    messages: list[ChatMessage],
    special_user_ids: tuple[int, ...] = (),
    owner_user_ids: tuple[int, ...] = (),
) -> list[dict[str, str]]:
    """Convert stored Discord messages into alternating user/assistant chat turns."""
    turns: list[dict[str, str]] = []
    special_set = set(special_user_ids)
    owner_set = set(owner_user_ids)

    for msg in messages:
        text = msg.content.strip()
        if not text:
            continue

        if msg.role == "assistant":
            if turns and turns[-1]["role"] == "assistant":
                turns[-1]["content"] = f"{turns[-1]['content']}\n{text}"
            elif turns:
                turns.append({"role": "assistant", "content": text})
            # Drop orphaned leading assistant messages so sliding window always starts on a user turn.
        else:
            speaker = (msg.author_name or "User").strip()
            if msg.author_id is not None and msg.author_id in special_set:
                speaker = f"{speaker} (special_person)"
            elif msg.author_id is not None and msg.author_id in owner_set:
                speaker = f"{speaker} (owner_irl)"
            line = f"[{speaker}]: {text}"
            if turns and turns[-1]["role"] == "user":
                turns[-1]["content"] = f"{turns[-1]['content']}\n{line}"
            else:
                turns.append({"role": "user", "content": line})

    return turns


class MemoryStore:
    def __init__(
        self,
        db_path: Path,
        max_history_per_channel: int = 200,
        ttl_hours: float = 0.0,
    ) -> None:
        self.db_path = db_path
        self.max_history_per_channel = max_history_per_channel
        self.ttl_hours = ttl_hours
        self._conn: aiosqlite.Connection | None = None

    async def init(self, initial_allowed_channels: tuple[int, ...] = ()) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            try:
                os.chmod(self.db_path.parent, 0o700)
            except OSError:
                pass
        self._conn = await aiosqlite.connect(self.db_path)
        if os.name == "posix":
            try:
                os.chmod(self.db_path, 0o600)
            except OSError:
                pass
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                author_id INTEGER,
                author_name TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_channel_id
            ON messages (channel_id, id DESC);

            CREATE INDEX IF NOT EXISTS idx_messages_created_at
            ON messages (created_at);

            CREATE TABLE IF NOT EXISTS allowed_channels (
                channel_id INTEGER PRIMARY KEY,
                added_by INTEGER,
                added_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS owner_style_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                normalized TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                source_channel_id INTEGER,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS owner_learned_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                normalized TEXT NOT NULL UNIQUE,
                rule TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS owner_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                file_path TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        await self._conn.commit()
        await self.prune_expired_messages()
        await self._sync_env_channels(initial_allowed_channels)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("MemoryStore is not initialized. Call await store.init() first.")
        return self._conn

    async def _sync_env_channels(self, env_channels: tuple[int, ...]) -> None:
        cur = await self.conn.execute(
            "SELECT value FROM bot_state WHERE key = 'seen_env_channels'"
        )
        row = await cur.fetchone()
        seen_ids: set[int] = set(json.loads(row["value"])) if row else set()

        now = datetime.now(UTC).isoformat()
        for cid in env_channels:
            if cid not in seen_ids:
                await self.conn.execute(
                    """
                    INSERT OR IGNORE INTO allowed_channels (channel_id, added_by, added_at)
                    VALUES (?, NULL, ?)
                    """,
                    (cid, now),
                )

        # Keep only currently present .env IDs in seen_env_channels so removing and
        # later re-adding an ID in .env works as expected, while Discord /channel_remove
        # persists across restarts as long as .env doesn't change.
        updated_seen = sorted(set(env_channels))
        await self.conn.execute(
            """
            INSERT INTO bot_state (key, value) VALUES ('seen_env_channels', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (json.dumps(updated_seen),),
        )
        await self.conn.commit()

    async def prune_expired_messages(self) -> int:
        if self.ttl_hours <= 0:
            return 0
        cutoff = (datetime.now(UTC) - timedelta(hours=self.ttl_hours)).isoformat(
            timespec="microseconds"
        )
        cur = await self.conn.execute(
            "DELETE FROM messages WHERE created_at < ?",
            (cutoff,),
        )
        deleted = cur.rowcount or 0
        if deleted > 0:
            await self.conn.commit()
            await self.conn.execute("VACUUM")
        return deleted

    async def add_message(
        self,
        channel_id: int,
        role: str,
        content: str,
        author_id: int | None = None,
        author_name: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat(timespec="microseconds")
        await self.conn.execute(
            """
            INSERT INTO messages (channel_id, role, author_id, author_name, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (channel_id, role, author_id, author_name, content, now),
        )
        # Prune older messages for this channel beyond retention cap.
        await self.conn.execute(
            """
            DELETE FROM messages
            WHERE channel_id = ?
              AND id NOT IN (
                  SELECT id FROM messages
                  WHERE channel_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (channel_id, channel_id, self.max_history_per_channel),
        )
        if self.ttl_hours > 0:
            cutoff = (datetime.now(UTC) - timedelta(hours=self.ttl_hours)).isoformat(
                timespec="microseconds"
            )
            await self.conn.execute(
                "DELETE FROM messages WHERE created_at < ?",
                (cutoff,),
            )
        await self.conn.commit()

    async def get_recent_messages(self, channel_id: int, limit: int) -> list[ChatMessage]:
        if self.ttl_hours > 0:
            await self.prune_expired_messages()
        cur = await self.conn.execute(
            """
            SELECT id, channel_id, role, author_id, author_name, content, created_at
            FROM messages
            WHERE channel_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (channel_id, limit),
        )
        rows = await cur.fetchall()
        return [
            ChatMessage(
                id=row["id"],
                channel_id=row["channel_id"],
                role=row["role"],
                author_id=row["author_id"],
                author_name=row["author_name"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]

    async def clear_channel_memory(self, channel_id: int) -> int:
        cur = await self.conn.execute(
            "DELETE FROM messages WHERE channel_id = ?", (channel_id,)
        )
        deleted = cur.rowcount or 0
        await self.conn.commit()
        if deleted > 0:
            await self.conn.execute("VACUUM")
        return deleted

    async def clear_all_memory(self) -> int:
        cur = await self.conn.execute("DELETE FROM messages")
        deleted = cur.rowcount or 0
        await self.conn.commit()
        if deleted > 0:
            await self.conn.execute("VACUUM")
        return deleted

    async def is_channel_allowed(self, channel_id: int) -> bool:
        cur = await self.conn.execute(
            "SELECT 1 FROM allowed_channels WHERE channel_id = ?", (channel_id,)
        )
        return (await cur.fetchone()) is not None

    async def add_allowed_channel(
        self, channel_id: int, added_by: int | None = None
    ) -> bool:
        now = datetime.now(UTC).isoformat()
        cur = await self.conn.execute(
            """
            INSERT OR IGNORE INTO allowed_channels (channel_id, added_by, added_at)
            VALUES (?, ?, ?)
            """,
            (channel_id, added_by, now),
        )
        await self.conn.commit()
        return (cur.rowcount or 0) > 0

    async def remove_allowed_channel(self, channel_id: int) -> bool:
        cur = await self.conn.execute(
            "DELETE FROM allowed_channels WHERE channel_id = ?", (channel_id,)
        )
        await self.conn.commit()
        return (cur.rowcount or 0) > 0

    async def list_allowed_channels(self) -> list[int]:
        cur = await self.conn.execute(
            "SELECT channel_id FROM allowed_channels ORDER BY added_at ASC"
        )
        rows = await cur.fetchall()
        return [int(row["channel_id"]) for row in rows]

    async def is_paused(self) -> bool:
        cur = await self.conn.execute(
            "SELECT value FROM bot_state WHERE key = 'paused'"
        )
        row = await cur.fetchone()
        return row is not None and row["value"] == "1"

    async def set_paused(self, paused: bool) -> None:
        await self.conn.execute(
            """
            INSERT INTO bot_state (key, value) VALUES ('paused', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            ("1" if paused else "0",),
        )
        await self.conn.commit()

    async def add_style_sample(
        self,
        content: str,
        source_channel_id: int | None = None,
        max_samples: int = 60,
    ) -> bool:
        cleaned = " ".join(content.split())
        if not cleaned:
            return False
        normalized = cleaned.lower()
        cur = await self.conn.execute(
            "SELECT 1 FROM owner_style_samples WHERE normalized = ?", (normalized,)
        )
        is_new = (await cur.fetchone()) is None
        now = datetime.now(UTC).isoformat(timespec="microseconds")
        await self.conn.execute(
            """
            INSERT INTO owner_style_samples (normalized, content, source_channel_id, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(normalized) DO UPDATE SET
                content = excluded.content,
                source_channel_id = excluded.source_channel_id,
                created_at = excluded.created_at
            """,
            (normalized, cleaned, source_channel_id, now),
        )
        await self.conn.execute(
            """
            DELETE FROM owner_style_samples
            WHERE id NOT IN (
                SELECT id FROM owner_style_samples
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            )
            """,
            (max_samples,),
        )
        await self.conn.commit()
        return is_new

    async def get_style_samples(self, limit: int = 20) -> list[str]:
        cur = await self.conn.execute(
            """
            SELECT content FROM owner_style_samples
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()
        return [str(row["content"]) for row in reversed(rows)]

    async def clear_style_samples(self) -> int:
        cur = await self.conn.execute("DELETE FROM owner_style_samples")
        deleted = cur.rowcount or 0
        await self.conn.commit()
        if deleted > 0:
            await self.conn.execute("VACUUM")
        return deleted

    async def add_learned_rule(self, rule: str, max_rules: int = 30) -> bool:
        cleaned = " ".join(rule.split())
        if not cleaned:
            return False
        normalized = cleaned.lower()
        cur = await self.conn.execute(
            "SELECT 1 FROM owner_learned_rules WHERE normalized = ?", (normalized,)
        )
        is_new = (await cur.fetchone()) is None
        now = datetime.now(UTC).isoformat(timespec="microseconds")
        await self.conn.execute(
            """
            INSERT INTO owner_learned_rules (normalized, rule, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(normalized) DO UPDATE SET
                rule = excluded.rule,
                created_at = excluded.created_at
            """,
            (normalized, cleaned, now),
        )
        await self.conn.execute(
            """
            DELETE FROM owner_learned_rules
            WHERE id NOT IN (
                SELECT id FROM owner_learned_rules
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            )
            """,
            (max_rules,),
        )
        await self.conn.commit()
        return is_new

    async def get_learned_rules(self, limit: int = 20) -> list[str]:
        cur = await self.conn.execute(
            """
            SELECT rule FROM owner_learned_rules
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()
        return [str(row["rule"]) for row in reversed(rows)]

    async def clear_learned_rules(self) -> int:
        cur = await self.conn.execute("DELETE FROM owner_learned_rules")
        deleted = cur.rowcount or 0
        await self.conn.commit()
        if deleted > 0:
            await self.conn.execute("VACUUM")
        return deleted

    async def count_style_memory(self) -> tuple[int, int]:
        cur_s = await self.conn.execute("SELECT COUNT(*) FROM owner_style_samples")
        samples_count = int((await cur_s.fetchone())[0])
        cur_r = await self.conn.execute("SELECT COUNT(*) FROM owner_learned_rules")
        rules_count = int((await cur_r.fetchone())[0])
        return samples_count, rules_count

    async def add_share_item(
        self,
        label: str,
        content: str = "",
        file_path: str = "",
        slug: str | None = None,
    ) -> SavedShareItem:
        clean_label = " ".join(label.split())
        if not clean_label:
            raise ValueError("Label cannot be empty.")
        clean_content = content.strip()
        clean_file_path = file_path.strip()
        if not clean_content and not clean_file_path:
            raise ValueError("Either content/link or file_path must be provided.")

        final_slug = (slug or make_share_slug(clean_label)).lower()
        now = datetime.now(UTC).isoformat(timespec="microseconds")
        await self.conn.execute(
            """
            INSERT INTO owner_links (slug, label, content, file_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                label = excluded.label,
                content = excluded.content,
                file_path = excluded.file_path,
                created_at = excluded.created_at
            """,
            (final_slug, clean_label, clean_content, clean_file_path, now),
        )
        await self.conn.commit()
        cur = await self.conn.execute(
            "SELECT id, slug, label, content, file_path, created_at FROM owner_links WHERE slug = ?",
            (final_slug,),
        )
        row = await cur.fetchone()
        assert row is not None
        return SavedShareItem(
            id=int(row["id"]),
            slug=str(row["slug"]),
            label=str(row["label"]),
            content=str(row["content"]),
            file_path=str(row["file_path"]),
            created_at=str(row["created_at"]),
        )

    async def list_share_items(self, limit: int = 40) -> list[SavedShareItem]:
        cur = await self.conn.execute(
            """
            SELECT id, slug, label, content, file_path, created_at
            FROM owner_links
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()
        return [
            SavedShareItem(
                id=int(row["id"]),
                slug=str(row["slug"]),
                label=str(row["label"]),
                content=str(row["content"]),
                file_path=str(row["file_path"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    async def remove_share_item(self, name_or_slug: str) -> SavedShareItem | None:
        cleaned = " ".join(name_or_slug.split()).lower()
        if not cleaned:
            return None
        candidate_slug = make_share_slug(cleaned)
        cur = await self.conn.execute(
            """
            SELECT id, slug, label, content, file_path, created_at
            FROM owner_links
            WHERE LOWER(slug) = ? OR LOWER(slug) = ? OR LOWER(label) = ?
            LIMIT 1
            """,
            (cleaned, candidate_slug, cleaned),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        item = SavedShareItem(
            id=int(row["id"]),
            slug=str(row["slug"]),
            label=str(row["label"]),
            content=str(row["content"]),
            file_path=str(row["file_path"]),
            created_at=str(row["created_at"]),
        )
        await self.conn.execute("DELETE FROM owner_links WHERE id = ?", (item.id,))
        await self.conn.commit()
        return item

    async def clear_share_items(self) -> list[SavedShareItem]:
        items = await self.list_share_items(limit=500)
        if items:
            await self.conn.execute("DELETE FROM owner_links")
            await self.conn.commit()
            await self.conn.execute("VACUUM")
        return items

