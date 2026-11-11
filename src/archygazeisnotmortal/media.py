from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Protocol

MAX_SHARE_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_ITEMS_PER_REPLY = 3

_SEND_TAG_RE = re.compile(r"\[\[send:([^\[\]\n]{1,80})\]\]", re.IGNORECASE)
_SONG_LINK_TAG_RE = re.compile(r"\[\[song_link:([^\[\]\n]{1,120})\]\]", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z']+")
_YT_VIDEO_ID_RE = re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"')
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

_ENGLISH_HINTS = frozenset(
    {
        "the",
        "is",
        "are",
        "was",
        "were",
        "im",
        "i'm",
        "ive",
        "i've",
        "you",
        "your",
        "you're",
        "my",
        "me",
        "we",
        "they",
        "what",
        "who",
        "who's",
        "when",
        "where",
        "why",
        "how",
        "have",
        "having",
        "again",
        "with",
        "this",
        "that",
        "from",
        "dont",
        "don't",
        "cant",
        "can't",
        "want",
        "need",
        "just",
        "bro",
        "dude",
        "please",
        "sorry",
        "thanks",
        "thank",
        "headache",
        "headaches",
        "playlist",
        "singer",
        "creator",
        "movie",
        "song",
        "send",
        "link",
        "shut",
    }
)

_INDONESIAN_HINTS = frozenset(
    {
        "yang",
        "yg",
        "ini",
        "itu",
        "apa",
        "apakah",
        "mana",
        "siapa",
        "kapan",
        "kenapa",
        "knp",
        "gimana",
        "gmn",
        "gw",
        "gue",
        "gua",
        "lu",
        "lo",
        "aku",
        "kamu",
        "km",
        "bgt",
        "banget",
        "ga",
        "gak",
        "nggak",
        "enggak",
        "tdk",
        "tidak",
        "iya",
        "ya",
        "aja",
        "doang",
        "sih",
        "nih",
        "tuh",
        "deh",
        "dong",
        "lagi",
        "lg",
        "sudah",
        "udah",
        "udh",
        "belum",
        "blm",
        "mau",
        "bisa",
        "sama",
        "sm",
        "di",
        "dari",
        "dr",
        "buat",
        "bt",
        "untuk",
        "utk",
        "kalau",
        "kalo",
        "karena",
        "krn",
        "tapi",
        "tp",
        "jadi",
        "jd",
        "terus",
        "trs",
        "biar",
        "emang",
        "emg",
        "beneran",
        "seriusan",
        "sakit",
        "kepala",
        "naon",
    }
)


class ShareItemLike(Protocol):
    slug: str
    label: str
    content: str
    file_path: str


def detect_obvious_language(text: str) -> str | None:
    """Return 'English', 'Indonesian', or None when short/ambiguous."""
    words = {w.lower() for w in _WORD_RE.findall(text)}
    if not words:
        return None
    en_hits = len(words & _ENGLISH_HINTS)
    id_hits = len(words & _INDONESIAN_HINTS)
    if en_hits >= 1 and id_hits == 0 and (en_hits >= 2 or len(words) <= 3):
        return "English"
    if id_hits >= 1 and en_hits == 0:
        return "Indonesian"
    if id_hits >= 2 and id_hits >= en_hits * 2:
        return "Indonesian"
    if en_hits >= 2 and en_hits >= id_hits * 2:
        return "English"
    return None


def sanitize_media_filename(raw_name: str, slug: str) -> str:
    base = Path(raw_name or "file").name
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "file"
    return f"{slug}_{cleaned}"[:120]


def _search_youtube_sync(query: str, timeout: float = 5.0) -> str | None:
    encoded = urllib.parse.quote(query)
    req = urllib.request.Request(
        f"https://www.youtube.com/results?search_query={encoded}",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", "ignore")
    match = _YT_VIDEO_ID_RE.search(html)
    if not match:
        return None
    return f"https://youtu.be/{match.group(1)}"


def _search_itunes_sync(query: str, timeout: float = 5.0) -> str | None:
    encoded = urllib.parse.quote(query)
    req = urllib.request.Request(
        f"https://itunes.apple.com/search?term={encoded}&entity=song&limit=1",
        headers={"User-Agent": "archygazeisnotmortal/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "ignore"))
    results = data.get("results") or []
    if not results:
        return None
    url = str(results[0].get("trackViewUrl") or "").strip()
    return url or None


async def lookup_music_link(query: str, timeout: float = 5.0) -> str | None:
    """Resolve a track query into a real playable YouTube or Apple Music link."""
    cleaned = " ".join(query.split())
    if not cleaned:
        return None
    for search_fn in (_search_youtube_sync, _search_itunes_sync):
        try:
            url = await asyncio.wait_for(
                asyncio.to_thread(search_fn, cleaned, timeout),
                timeout=timeout + 1.0,
            )
            if url:
                return url
        except Exception:
            continue
    return None


async def resolve_reply_media(
    text: str,
    saved_items: Sequence[ShareItemLike],
    media_dir: Path,
    lookup_fn: Callable[[str], Awaitable[str | None]] | None = None,
) -> tuple[str, list[Path]]:
    """Resolve [[send:slug]] attachments and [[song_link:query]] placeholders in reply text."""
    if not text:
        return "", []
    music_lookup = lookup_fn or lookup_music_link

    by_slug = {item.slug.lower(): item for item in saved_items}
    resolved_media_root = media_dir.resolve()
    attachment_paths: list[Path] = []
    seen_paths: set[Path] = set()

    def _replace_send(match: re.Match[str]) -> str:
        slug = match.group(1).strip().lower()
        item = by_slug.get(slug)
        if item is not None and item.file_path and len(attachment_paths) < MAX_ITEMS_PER_REPLY:
            candidate = Path(item.file_path).resolve()
            if (
                candidate.is_relative_to(resolved_media_root)
                and candidate.is_file()
                and candidate not in seen_paths
            ):
                seen_paths.add(candidate)
                attachment_paths.append(candidate)
                return ""
        if item is not None and item.content:
            return item.content
        return ""

    updated = _SEND_TAG_RE.sub(_replace_send, text)

    queries: list[str] = []
    for match in _SONG_LINK_TAG_RE.finditer(updated):
        q = " ".join(match.group(1).split())
        if q and q not in queries and len(queries) < MAX_ITEMS_PER_REPLY:
            queries.append(q)

    resolved_urls: dict[str, str | None] = {}
    if queries:
        results = await asyncio.gather(
            *(music_lookup(q) for q in queries),
            return_exceptions=True,
        )
        for q, result in zip(queries, results, strict=True):
            resolved_urls[q] = result if isinstance(result, str) and result else None

    def _replace_song_link(match: re.Match[str]) -> str:
        q = " ".join(match.group(1).split())
        return resolved_urls.get(q) or ""

    updated = _SONG_LINK_TAG_RE.sub(_replace_song_link, updated)
    updated = re.sub(r"[ \t]+\n", "\n", updated)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    updated = re.sub(r"[ \t]{2,}", " ", updated).strip()
    return updated, attachment_paths
