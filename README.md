# archygazeisnotmortal

An async Python (`discord.py`) text bot that represents **you** in designated Discord channels — with an editable persona profile (`persona.md`), persistent SQLite conversation memory, message debouncing/batching, and multi-provider free AI support.

## Features
- **Auto-replies in designated channels** (configured via `.env` or live in Discord with `/channel_add` & `/channel_remove`)
- **Persona profile (`persona.md`)** with bio, texting tone, rules, boundaries, and example conversations — hot-reloadable via `/persona_reload`
- **Persistent SQLite memory (`data/memory.db`)** per channel/DM with multi-speaker awareness (`[Alex]: ...`)
- **Message debouncing & cooldowns** so rapid consecutive messages (`"yo"`, `"u there?"`, `"hop on"`) combine into one natural reply without wasting free API quota
- **Multi-provider free AI support + automatic fallback**:
  - `groq` (fast free developer tier)
  - `gemini` (Google AI Studio free tier)
  - `openrouter` (`openrouter/free` community-hosted free models)
  - `ollama` (100% free forever, runs locally on your own GPU/CPU)

## Quick Setup

### 1. Create your Discord Bot
1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application**.
2. Open the **Bot** tab -> click **Reset Token** and save the token.
3. Scroll down to **Privileged Gateway Intents** and enable **Message Content Intent**, then click **Save Changes**.
4. Open **OAuth2 -> URL Generator**:
   - Scopes: `bot`, `applications.commands`
   - Bot Permissions: `View Channels`, `Send Messages`, `Read Message History`, `Add Reactions`
5. Open the generated invite URL to add the bot to your server.

### 2. Configure `.env`
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Fill in:
- `DISCORD_TOKEN`
- `OWNER_USER_IDS` (enable Discord Developer Mode -> right-click yourself -> Copy User ID)
- `AI_PROVIDER` (`groq`, `gemini`, `openrouter`, or `ollama`) + its corresponding API key

> **Tip:** You can also set `FALLBACK_AI_PROVIDER=openrouter` (with `OPENROUTER_API_KEY`) so if your primary free API hits a rate limit (`429`), the bot automatically switches to the backup provider.

### 3. Customize `persona.md`
Copy `persona.example.md` to `persona.md` (already git-ignored for privacy) and edit it with your name, hobbies, texting style, boundaries, and a few sample chats so it sounds like you. (On cloud hosts without local file edits, you can instead set `PERSONA_CONTENT` as a secret environment variable.)

### 4. Run the Bot
```powershell
uv run archygazeisnotmortal
```
(Or using the project virtualenv directly: `.venv\Scripts\python.exe -m archygazeisnotmortal`)

## Slash Commands
- `/channel_add [channel]` — Enable auto-replies in a channel
- `/channel_remove [channel]` — Disable auto-replies in a channel
- `/channel_list` — List active auto-reply channels
- `/persona_reload` — Reload `persona.md` without restarting
- `/memory_clear` — Wipe conversation memory in the current channel or DM
- `/memory_clear_all` — Wipe all saved conversation memory across every channel and DM
- `/teach <rule>` — Teach the bot a permanent style/behavior rule or correction
- `/style_sync [channel] [limit]` — Scan recent messages from `OWNER_USER_IDS` to learn your texting voice
- `/style_add <example>` — Manually save a short sample line of how you talk
- `/style_list` / `/style_clear` — Inspect or clear saved voice samples and learned rules
- `/bot_pause` / `/bot_resume` — Pause or resume auto-replies
- `/bot_status` — Check active provider, fallback, and channels
- `/ask <message>` — Send a one-off message to your persona in any channel

> **Prefix tip:** Messages starting with `//` (configurable via `IGNORE_PREFIX`) are ignored by the bot so you can chat in an auto-reply channel without triggering it.

## 24/7 Private Deployment

GitHub stores your code (not a running Discord bot), so keep your repo **private** and deploy it to an always-on host or VPS:

1. **Private files stay local:** `.gitignore` excludes `.env`, `persona.md`, and `data/` (`*.db*`), so your tokens, persona, and conversation history are never pushed to GitHub.
2. **Containerized build:** A `Dockerfile` is included for hosts like Railway, Fly.io, Koyeb, Render (Background Worker), or any Docker VPS.
3. **Environment variables on your host:** Set `DISCORD_TOKEN`, `OWNER_USER_IDS`, `AI_PROVIDER`, your API key, `MEMORY_TTL_HOURS=24`, and `PERSONA_CONTENT` (your `persona.md` text) as hidden secrets on the host, and attach a persistent volume to `/app/data` if you want memory and `/channel_add` settings to survive restarts. If your host injects `PORT` (e.g., Render/Koyeb web services), the bot automatically responds `200 OK` on `/` and `/health`.

