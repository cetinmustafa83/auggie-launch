# auggie-launch

Launcher for the installed `auggie` / Augment Code CLI that starts a local
**Python** Augment-compatible proxy and forwards chat traffic to a standard
OpenAI-compatible Chat Completions endpoint.

**No secrets or private endpoints are hard-coded.** Configure upstream model
settings through `.env` or environment variables.

## Copyright / 免责声明

Copyright (c) 2026 auggie-launch contributors

**仅供学习与研究使用；其他用途后果自负。**

This project is for learning and research only. Any other use is at your own
risk. See [COPYRIGHT](COPYRIGHT).

## Key default: RAG/indexing disabled

`auggie-launch` defaults to:

```env
AUGGIE_LAUNCH_INDEXING_MODE=complete
```

That means the local proxy tells Auggie:

- `/find-missing`: no memory/blob objects are missing
- `/batch-upload`: upload requests are acknowledged without storing content

This prevents Auggie from uploading or building RAG/indexing material by
default. The proxy project in `/root/projects/auggieproxy` was used only as a
protocol reference; `auggie-launch` runs its own Python proxy and does not depend
on Deno at runtime.

## What it does

1. Loads config from env / `.env`.
2. Starts a local Python HTTP proxy on `127.0.0.1`.
3. Injects Augment session environment variables:
   - `AUGMENT_API_URL`
   - `AUGMENT_API_TOKEN`
   - `AUGMENT_SESSION_AUTH`
4. Executes the already-installed `auggie` binary with your original args.
5. Handles Augment login/model/settings/indexing endpoints locally.
6. Translates chat/completion requests to `POST {base}/chat/completions`.

```text
+-------------+              +-----------------------+              +--------------+
|             | Augment API  |                       | Chat API     |              |
| auggie CLI  | -----------> | auggie-launch Python  | -----------> | Upstream LLM |
|             | local URL    | local proxy           | OpenAI-style |              |
+-------------+              +-----------------------+              +--------------+
```

## Quick start

```bash
# 1. Install wrapper in ~/.local/bin and create user config template
./install.sh



# 2. Edit user config
$EDITOR ~/.config/auggie-launch/.env

# 3. Ensure PATH and run
export PATH="$HOME/.local/bin:$PATH"
auggie-launch
auggie-launch --print "hello"
```
`./install.sh` also auto-installs the upstream CLI when it is missing (use `--skip-cli` to opt out).

Project-local config is supported:

```bash
cp .env.example .env
# edit .env; this file is gitignored
auggie-launch --print "hello"
```

## Configuration

### Required

| Variable | Meaning |
|----------|---------|
| `AUGGIE_LAUNCH_BASE_URL` | OpenAI-compatible base URL, e.g. `https://gateway.example/v1` |
| `AUGGIE_LAUNCH_MODEL` | Real model name sent to `/chat/completions` |
| `AUGGIE_LAUNCH_API_KEY` | Bearer token for the upstream gateway |

### Optional

| Variable | Meaning |
|----------|---------|
| `AUGGIE_LAUNCH_API_KEYS` | Comma-separated key rotation list; overrides single key when set |
| `AUGGIE_LAUNCH_PORT` | Local Python proxy port; default `0` means auto-pick free port |
| `AUGGIE_LAUNCH_INDEXING_MODE` | `complete` by default; disables RAG/blob upload path |
| `AUGGIE_LAUNCH_USER_AGENT` | Upstream user-agent; default `codex-cli` |
| `AUGGIE_LAUNCH_UPSTREAM_APP_NAME` | Upstream app metadata; default `Codex` |
| `AUGGIE_LAUNCH_SANITIZE_UPSTREAM_PROMPTS` | Reserved prompt sanitization toggle; default `false` |
| `AUGGIE_LAUNCH_LOCAL_TOKEN` | Dummy local token between `auggie` and proxy |
| `AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS` | Model metadata exposed to Auggie; default `200000` |
| `AUGGIE_LAUNCH_MODEL_MAX_OUTPUT_TOKENS` | Model metadata exposed to Auggie; default `16000` |
| `AUGGIE_LAUNCH_REASONING_EFFORT` | Optional upstream reasoning effort: `low`, `medium`, or `high` |
| `AUGGIE_LAUNCH_SYSTEM_PROMPT` | Optional system prompt prepended upstream |
| `AUGGIE_LAUNCH_VERBOSE=1` | Print proxy diagnostics |
| `AUGGIE_LAUNCH_DEBUG_DIR` | Directory for debug request dumps |
| `AUGGIE_BIN` | Path to installed Auggie CLI; default `auggie` |

### `.env` load priority

1. `AUGGIE_LAUNCH_ENV` if set
2. Package directory `.env` (launcher-local, next to `main.py`)
3. `./.env` or `./.auggie-launch.env` (cwd)
4. Parent directories (up to 6 levels)
5. `~/.config/auggie-launch/.env`
6. `~/.auggie-launch.env`

For auggie-launch-managed keys (`AUGGIE_LAUNCH_*` and `AUGGIE_BIN`), higher-priority `.env` files override stale shell exports.

## Usage

Normal Auggie CLI args are passed through:

```bash
auggie-launch
auggie-launch --print "reply with ok"
auggie-launch --continue
auggie-launch --print --quiet "summarize this repo"
AUGGIE_LAUNCH_VERBOSE=1 auggie-launch --print "hello"
```

Launcher-specific options:

```bash
auggie-launch --print-env      # show resolved config without starting auggie
auggie-launch --proxy-only     # run only the Python proxy in foreground
auggie-launch --help
```

Use `--` if an Auggie argument could be confused with a launcher option:

```bash
auggie-launch -- --print "hello"
```

## install.sh

```bash
./install.sh              # install wrapper + create user .env from example
./install.sh --force-env  # overwrite user .env from .env.example
./install.sh --no-env     # only install wrapper
./install.sh --link       # symlink repo launcher instead of generated wrapper
./install.sh --bin-dir ~/bin --config-dir ~/.config/auggie-launch
```

The installer checks for:

- `python3`
- `auggie` on `PATH` (warning only)

Deno is not required.

## Implemented local Augment behavior

Handled locally:

- `POST /token`
- `POST /get-models`, `/models`, `/model-config`
- `POST /get-credit-info`
- `POST /get-billing-summary`
- `POST /find-missing`
- `POST /batch-upload`
- `POST /checkpoint-blobs`
- `settings/*`
- `tenant-secrets/*`, `user-secrets/*`
- `remote-agents/*`, `cloud-agents/*`, `agent-workspace/*`
- telemetry/feedback endpoints

Forwarded upstream:

- `POST /chat-stream`
- `POST /prompt-enhancer`
- `POST /chat`
- `POST /remote-agents/chat`
- `POST /completion`
- `POST /completion/request`
- `POST /completion/complete`
- `POST /chat-input-completion`

The current Python translator is intentionally compact. If Auggie exposes a new
request shape for complex tool-node streaming, inspect verbose logs and extend
`main.py` from the `auggieproxy` protocol notes.

## Layout

```text
auggie-launch/
  main.py              # standalone Python proxy + launcher
  auggie-launch        # small entrypoint that execs main.py
  install.sh           # install wrapper + user .env template
  .env.example         # public config template
  .gitignore           # ignores secrets and logs
  COPYRIGHT
  README.md
```

## Troubleshooting

### `error: missing required configuration`

Fill these in:

```env
AUGGIE_LAUNCH_BASE_URL=
AUGGIE_LAUNCH_MODEL=
AUGGIE_LAUNCH_API_KEY=
```

### Verify config without launching Auggie

```bash
auggie-launch --print-env
```

### Run proxy only

```bash
AUGGIE_LAUNCH_VERBOSE=1 auggie-launch --proxy-only
```

It prints the local proxy URL, then keeps serving until interrupted.

### `auggie` not found

Auggie is expected to already be installed. Either add it to `PATH` or set:

```env
AUGGIE_BIN=/absolute/path/to/auggie
```
