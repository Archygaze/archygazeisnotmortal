from __future__ import annotations

import logging
import sys

import discord

from archygazeisnotmortal.bot import PersonaBot
from archygazeisnotmortal.config import Settings


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger("archygazeisnotmortal")

    try:
        settings = Settings.from_env()
    except Exception as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    errors = settings.validate_for_runtime()
    if errors:
        for err in errors:
            logger.error("%s", err)
        sys.exit(1)

    if not settings.persona_file.exists() and not settings.persona_content:
        logger.error(
            "Persona missing: create %s (e.g. from persona.example.md) or set PERSONA_CONTENT in .env.",
            settings.persona_file,
        )
        sys.exit(1)

    bot = PersonaBot(settings)
    try:
        bot.run(settings.discord_token, log_handler=None)
    except discord.PrivilegedIntentsRequired:
        logger.error(
            "Discord rejected Privileged Intents (close code 4014). "
            "Open the Discord Developer Portal -> your App -> Bot -> enable 'Message Content Intent' -> Save Changes."
        )
        sys.exit(1)
    except discord.LoginFailure:
        logger.error("Invalid DISCORD_TOKEN. Double-check your token in .env.")
        sys.exit(1)


if __name__ == "__main__":
    main()

