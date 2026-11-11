from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite


@dataclass(frozen=True, slots=True)
class ChatMessage:
    id: int
    channel_id: int
    role: str
    author_id: int | None
    author_name: str | None
    content: str
    created_at: str


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

