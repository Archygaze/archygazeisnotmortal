from pathlib import Path
from types import SimpleNamespace

from archygazeisnotmortal.media import (
    detect_obvious_language,
    resolve_reply_media,
    sanitize_media_filename,
)


def test_detect_obvious_language() -> None:
    assert detect_obvious_language("im having headaches again") == "English"
    assert detect_obvious_language("who's your favourite singer") == "English"
    assert detect_obvious_language("ya mana") == "Indonesian"
    assert detect_obvious_language("naon sih ga jelas") == "Indonesian"
    assert detect_obvious_language("😭😭😭") is None


def test_sanitize_media_filename() -> None:
    assert sanitize_media_filename("../../weird name (1).png", "cat_pic") == "cat_pic_weird_name_1_.png"


async def test_resolve_reply_media_handles_files_song_links_and_path_traversal(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    safe_file = media_dir / "cat_pic_photo.png"
    safe_file.write_bytes(b"png-bytes")
    outside_secret = tmp_path / "secret.txt"
    outside_secret.write_text("top-secret", encoding="utf-8")

    saved_items = [
        SimpleNamespace(slug="cat_pic", label="cat pic", content="", file_path=str(safe_file)),
        SimpleNamespace(slug="escape", label="escape", content="", file_path=str(outside_secret)),
        SimpleNamespace(slug="playlist", label="chill playlist", content="https://open.spotify.com/playlist/abc", file_path=""),
    ]

    async def fake_lookup(query: str) -> str | None:
        if "Rhubarb" in query:
            return "https://youtu.be/_AWIqXzvX-U"
        return None

    reply_text, paths = await resolve_reply_media(
        "here you go [[send:cat_pic]] [[send:escape]]\n"
        "try this [[song_link: Aphex Twin - Rhubarb]]\n"
        "missing [[song_link: nonexistent_track]]",
        saved_items=saved_items,
        media_dir=media_dir,
        lookup_fn=fake_lookup,
    )
    assert paths == [safe_file.resolve()]
    assert "https://youtu.be/_AWIqXzvX-U" in reply_text
    assert "[[send:" not in reply_text
    assert "[[song_link:" not in reply_text
