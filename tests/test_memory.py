from datetime import UTC, datetime, timedelta
from pathlib import Path
import pytest

from archygazeisnotmortal.memory import ChatMessage, MemoryStore, format_messages_for_llm


async def test_memory_store_messages_and_window(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db", max_history_per_channel=5)
    await store.init()
    try:
        for i in range(7):
            await store.add_message(
                channel_id=100,
                role="user",
                content=f"msg {i}",
                author_id=1,
                author_name="Alex",
            )
        recent = await store.get_recent_messages(channel_id=100, limit=10)
        # Retention cap kept only last 5
        assert [m.content for m in recent] == ["msg 2", "msg 3", "msg 4", "msg 5", "msg 6"]

        deleted = await store.clear_channel_memory(100)
        assert deleted == 5
        assert await store.get_recent_messages(channel_id=100, limit=10) == []
    finally:
        await store.close()


async def test_memory_store_env_channel_sync_respects_runtime_removal(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"

    # First startup seeds channels from .env
    store1 = MemoryStore(db_path)
    await store1.init(initial_allowed_channels=(10, 20))
    assert await store1.list_allowed_channels() == [10, 20]
    await store1.remove_allowed_channel(10)
    assert await store1.list_allowed_channels() == [20]
    await store1.close()

    # Second startup with unchanged .env keeps channel 10 removed
    store2 = MemoryStore(db_path)
    await store2.init(initial_allowed_channels=(10, 20))
    assert await store2.list_allowed_channels() == [20]
    await store2.close()

    # Third startup with a newly added channel 30 in .env seeds 30, keeps 10 removed
    store3 = MemoryStore(db_path)
    await store3.init(initial_allowed_channels=(10, 20, 30))
    assert await store3.list_allowed_channels() == [20, 30]
    await store3.close()


async def test_memory_store_pause_flag(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    await store.init()
    try:
        assert await store.is_paused() is False
        await store.set_paused(True)
        assert await store.is_paused() is True
        await store.set_paused(False)
        assert await store.is_paused() is False
    finally:
        await store.close()


def test_format_messages_for_llm_coalesces_turns_and_drops_leading_assistant() -> None:
    msgs = [
        ChatMessage(1, 1, "assistant", 99, "Me", "orphan reply", "t1"),
        ChatMessage(2, 1, "user", 1, "Alex", "yo", "t2"),
        ChatMessage(3, 1, "user", 2, "Sam", "sup guys", "t3"),
        ChatMessage(4, 1, "assistant", 99, "Me", "nm just chilling", "t4"),
        ChatMessage(5, 1, "user", 1, "Alex", "hop on", "t5"),
    ]
    formatted = format_messages_for_llm(msgs, special_user_ids=(2,))
    assert formatted == [
        {"role": "user", "content": "[Alex]: yo\n[Sam (special_person)]: sup guys"},
        {"role": "assistant", "content": "nm just chilling"},
        {"role": "user", "content": "[Alex]: hop on"},
    ]


async def test_memory_store_ttl_expiration_and_clear_all(tmp_path: Path) -> None:
    db_path = tmp_path / "ttl.db"
    store = MemoryStore(db_path, max_history_per_channel=20, ttl_hours=2.0)
    await store.init(initial_allowed_channels=(101,))
    try:
        old_ts = (datetime.now(UTC) - timedelta(hours=5)).isoformat(timespec="microseconds")
        await store.conn.execute(
            """
            INSERT INTO messages (channel_id, role, author_id, author_name, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (101, "user", 1, "Alex", "old secret", old_ts),
        )
        await store.conn.commit()
        await store.add_message(channel_id=101, role="user", content="fresh msg", author_id=1, author_name="Alex")
        await store.add_message(channel_id=202, role="user", content="dm msg", author_id=2, author_name="Sam")

        recent_101 = await store.get_recent_messages(101, limit=10)
        assert [m.content for m in recent_101] == ["fresh msg"]

        await store.set_paused(True)
        wiped = await store.clear_all_memory()
        assert wiped == 2
        assert await store.get_recent_messages(101, limit=10) == []
        assert await store.get_recent_messages(202, limit=10) == []
        # Allowed channels & pause state remain intact after wiping messages
        assert await store.list_allowed_channels() == [101]
        assert await store.is_paused() is True
    finally:
        await store.close()

