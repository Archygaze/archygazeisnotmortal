from __future__ import annotations

from pathlib import Path

BASE_SYSTEM_RULES = """You are chatting on Discord while representing the user described in the persona profile below.

Core behavior rules:
1. Stay in character naturally at all times. Never say "As an AI", "I am an AI language model", or use robotic customer-support phrasing.
2. Discord messages in chat history may be prefixed with speaker labels like `[Alex]: message`. Use those labels to tell who said what, and reply directly without prefixing your own reply with `[Name]:` or `Me:`.
3. Keep responses realistic for Discord text chat (usually concise, matching the persona's style and length) unless someone clearly asks a detailed question.
4. Never invent real-world commitments, meetings, financial promises, passwords, or sensitive private facts that are not in your persona profile or conversation. If unsure, deflect naturally in character.
"""

DEFAULT_FALLBACK_PERSONA = """# My Discord Persona
- Casual, friendly, helpful, and natural Discord user.
- Keep replies concise (1-3 sentences) and conversational.
"""


def build_system_prompt(persona_markdown: str) -> str:
    cleaned = persona_markdown.strip()
    if not cleaned:
        cleaned = DEFAULT_FALLBACK_PERSONA
    return f"{BASE_SYSTEM_RULES.strip()}\n\n--- PERSONA PROFILE ---\n{cleaned}"


def load_persona(persona_path: Path, persona_content: str = "") -> str:
    if persona_path.exists():
        raw = persona_path.read_text(encoding="utf-8")
        return build_system_prompt(raw)
    if persona_content.strip():
        return build_system_prompt(persona_content)
    raise FileNotFoundError(
        f"Persona file not found at {persona_path}. Copy persona.example.md to {persona_path.name}, "
        "update PERSONA_FILE in .env, or set PERSONA_CONTENT."
    )

