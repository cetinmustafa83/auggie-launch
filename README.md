# auggie-launch

High-performance, modern launcher for the installed `auggie` / Augment Code CLI that starts a local
**Python** Augment-compatible proxy and forwards chat traffic to standard OpenAI-compatible endpoints,
with **first-class, native support for 9router**, Anthropic Claude 3.7 (thinking mode), DeepSeek-R1, and OpenAI o-series models.

**No secrets or private endpoints are hard-coded.** Configure upstream model
settings through `.env` or environment variables.



---

## Deep 9router Local Integration

### Zero-Config Auto-Discovery

On startup the launcher automatically reads:

- `~/.9router/db.json` — combos, model aliases, provider connections, API keys, tunnel URL, Caveman settings
- `~/.9router/model-catalog.json` — model capabilities (vision, audio, PDF, video) and context windows

No environment variables are required. 9router is detected when `AUGGIE_LAUNCH_BASE_URL`
targets port `20128` / a 9router host, or simply by finding the local DB.

### Dual Connection Guarantee

The health check (`--check`) performs:

1. **Local socket test** against `AUGGIE_LAUNCH_BASE_URL`
2. **Tunnel fallback** to the Cloudflare URL from `db.json` (`settings.tunnelUrl`) if the local socket fails

This keeps connectivity even if the local 9router process is temporarily down.

At **runtime** the proxy does the same: if an upstream request fails with a transport error
(local 9router died mid-session) and the tunnel answers, all further traffic — chat completions
and `/models` — fails over to the tunnel for the rest of the process lifetime.

### 9router Service Controller

| Command | Description |
|---|---|
| `--start-9router` | Starts 9router; installs it from npm first if the binary is missing |
| `--install-9router` | `npm i -g 9router@latest --prefer-online`, then seeds the DB if absent |
| `--update-9router` | Updates 9router to the latest npm release |
| `--restore-9router-db` | Restores `~/.9router/db.json` from the newest backup in `9router/db/` (existing DB kept as `db.json.bak`) |
| `--stats`, `--usage` | Queries `/api/usage` for live token savings, request counts, and provider breakdown |
| `--combos` | Lists 9router combos and their fallback model groups |

### Self-Healing Install

If 9router is targeted but not detected on the system, the launcher installs it automatically
(`npm i -g 9router@latest --prefer-online`) and restores `~/.9router/db.json` from the newest
backup in the repo's `9router/db/` folder when no live DB exists. Disable with
`AUGGIE_LAUNCH_AUTO_INSTALL_9ROUTER=false`.

> The bundled DB backup holds live API keys, so `9router/` is git-ignored — keep it local.

### Full Auggie CLI Injection

The launcher injects four layers of configuration into Auggie:

1. **Environment variables** — standard OpenAI/Anthropic endpoints plus every active
   9router provider key (`TAVILY_API_KEY`, `FIRECRAWL_API_KEY`, `MINIMAX_API_KEY`,
   `JINA_READER_API_KEY`, …, with a generic `{PROVIDER}_API_KEY` fallback).
2. **Dynamic model registry** — all combos, aliases, and live upstream models, each with
   per-model context limits and capabilities taken from the catalog.
3. **Feature flags** — reasoning/thinking streaming, indexing mode, completion-token negotiation.
4. **MCP tools** — an auto-generated MCP config when provider keys exist: Tavily (web search),
   Firecrawl (web scraping), Sequential Thinking (structured reasoning).

### Verification Commands

```bash
python3 main.py --check      # health + 9router diagnostics (local socket, then tunnel)
python3 main.py --models     # combos, aliases, and live upstream models
python3 main.py --combos     # combos with their fallback chains
python3 main.py --stats      # token savings, providers, tunnel status
python3 main.py --print-env  # resolved configuration injected into Auggie
```

---

## Key Features & Modern Techniques
### 1. First-Class 9router Support (`localhost:20128`)
- **Auto-Detection**: Automatically detects 9router when `AUGGIE_LAUNCH_BASE_URL` targets port `20128` or matches 9router hosts.
- **Dynamic Model Discovery**: Queries 9router's `/v1/models` registry and automatically exposes all available models (`free`, `fast`, `claude-3-7-sonnet`, `deepseek-r1`, `gpt-4o`, etc.) directly to Auggie CLI with appropriate context limits.
- **3-Tier Fallback Observability**: Reads and displays 9router headers (`X-Router-Provider`, `X-Model-Used`, `X-Token-Savings`) in verbose logs.
- **Caveman Mode**: Optional `AUGGIE_LAUNCH_9ROUTER_CAVEMAN=true` to drastically reduce input/output token consumption.
- **Provider Routing**: Target specific 9router backends using `AUGGIE_LAUNCH_9ROUTER_PROVIDER=anthropic`.
- **Doctor Diagnostics**: Built-in CLI health check (`auggie-launch --check`) to verify socket connections, authentication, model latency, and CLI installation.

### 2. Advanced LLM Limits & Context Truncation
- **Turn-Atomic Truncation**: Unlike naive character truncation, messages are grouped into atomic conversational turns. **Assistant `tool_calls` and their corresponding `tool` results are NEVER separated or orphaned**, preventing `400 Bad Request: Invalid message sequence` errors.
- **Heuristic Token Estimator**: Structural token estimator tuned for code, JSON schemas, whitespace, and multi-byte characters.
- **Reasoning & Thinking Models**: Full streaming support for **DeepSeek-R1**, **Claude 3.7 Sonnet**, and **o1/o3** thinking blocks (`reasoning_content`, `thought`), formatted cleanly as `<think>...</think>` in Auggie streams.
- **Parameter Negotiation**: Auto-negotiates `max_completion_tokens` vs `max_tokens` for newer models, and automatically swaps parameters if upstream returns a 400 rejection.

### 3. Modern Proxy Performance & Networking
- **HTTP Connection Pooling**: Thread-safe persistent HTTP/1.1 connection pool with Keep-Alive, eliminating TLS/TCP handshake latency on every request.
- **Full Jitter Exponential Backoff**: Prevents thundering-herd spikes on rate limits (429) or transient 5xx gateway errors following the AWS architecture standard.
- **Streaming Keep-Alive & Cancellation**: Emits periodic heartbeats during long reasoning phases and immediately terminates upstream streaming if the client disconnects (Ctrl+C / socket closure).
- **Tool Call Merging & JSON Repair**: Reliably merges interleaved streaming tool call deltas by index and auto-repairs truncated JSON argument strings.

---

## Architecture

```text
+-------------+                  +---------------------------+                  +---------------------+
|             |  Augment API     |                           |  Chat API (v1)   |                     |
| auggie CLI  | ---------------> |  auggie-launch Python     | ---------------> | 9router AI Gateway  |
|             |  127.0.0.1:port  |  - Connection Pool        |  HTTP Keep-Alive | (localhost:20128)   |
+-------------+                  |  - Turn-Atomic Truncation |                  +----------+----------+
                                 |  - Reasoning Stream       |                             |
                                 |  - 9router Auto-Discovery |                             v
                                 +---------------------------+                  +---------------------+
                                                                                | Claude / DeepSeek / |
                                                                                | GPT / Free Tiers    |
                                                                                +---------------------+
```

---

## Quick Start

### 1. Configure with 9router (or any OpenAI gateway)

Create or edit `.env` in the repository or `~/.config/auggie-launch/.env`:

```env
AUGGIE_LAUNCH_BASE_URL=http://localhost:20128/v1
AUGGIE_LAUNCH_MODEL=free
AUGGIE_LAUNCH_API_KEY=your-9router-api-key
```

### 2. Run Doctor Health Check

Verify your setup with the new diagnostic tool:

```bash
python3 main.py --check
```

Example diagnostic output:
```text
============================================================
🔍 AUGGIE-LAUNCH HEALTH & 9ROUTER DIAGNOSTICS
============================================================
Target Base URL : http://localhost:20128/v1
Target Model    : free
API Key         : sk-561a404... (1 key(s) loaded)
9router Detected: YES (port 20128 or host match)
Indexing Mode   : complete
Reasoning Stream: True
------------------------------------------------------------
1. Testing connection to Upstream / 9router...
   [OK] Socket connection succeeded to localhost:20128
2. Querying Model Registry (/models)...
   [OK] Discovered 12 models available:
        - free
        - fast
        - claude-3-7-sonnet
        - deepseek-r1
        ... and 8 more.
3. Testing completion ping with model 'free'...
   [OK] Upstream completion response in 240.2ms: 'pong'
4. Checking Auggie CLI binary...
   [OK] Auggie binary found: auggie -> auggie 0.x.x
============================================================
✅ All systems ready for auggie-launch!
```

### 3. List Available Models

Query upstream/9router to view all available models:

```bash
python3 main.py --models
```

### 4. Run Auggie

```bash
./auggie-launch --print "hello"
```

---

## Configuration Reference

### Required Settings

| Variable | Description |
|---|---|
| `AUGGIE_LAUNCH_BASE_URL` | Upstream OpenAI-compatible URL (e.g. `http://localhost:20128/v1`) |
| `AUGGIE_LAUNCH_MODEL` | Target model name (e.g. `free`, `claude-3-7-sonnet`, `deepseek-r1`) |
| `AUGGIE_LAUNCH_API_KEY` | Bearer token / API key |

### 9router Options

| Variable | Default | Description |
|---|---|---|
| `AUGGIE_LAUNCH_FORCE_9ROUTER` | `false` | Force 9router mode even if port is not 20128 |
| `AUGGIE_LAUNCH_DYNAMIC_MODELS` | `true` | Auto-query `/v1/models` and register all models in Auggie |
| `AUGGIE_LAUNCH_9ROUTER_CAVEMAN` | `false` | Enable 9router Caveman mode (`X-Caveman-Mode: true`) |
| `AUGGIE_LAUNCH_9ROUTER_PROVIDER` | `""` | Target specific provider (`X-Router-Provider: ...`) |
| `AUGGIE_LAUNCH_AUTO_INSTALL_9ROUTER` | `true` | Auto-install 9router via npm when it is not detected |

### Modern LLM & Reasoning Settings

| Variable | Default | Description |
|---|---|---|
| `AUGGIE_LAUNCH_STREAM_THINKING` | `true` | Stream thinking tokens (`<think>...</think>`) to Auggie |
| `AUGGIE_LAUNCH_USE_COMPLETION_TOKENS` | `auto` | `auto`, `true`, or `false` for `max_completion_tokens` |
| `AUGGIE_LAUNCH_REASONING_EFFORT` | `""` | Optional reasoning effort: `low`, `medium`, `high` |
| `AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS`| `200000` | Context limit budget for truncation engine |
| `AUGGIE_LAUNCH_MODEL_MAX_OUTPUT_TOKENS` | `16000` | Reserved output tokens |

### Proxy & Networking Settings

| Variable | Default | Description |
|---|---|---|
| `AUGGIE_LAUNCH_CONNECTION_POOL` | `true` | Enable HTTP/1.1 Keep-Alive connection pooling |
| `AUGGIE_LAUNCH_UPSTREAM_RETRIES` | `2` | Number of retries on 429/5xx errors |
| `AUGGIE_LAUNCH_429_FREEZE_SECONDS` | `60.0` | Key cooldown on rate limits |
| `AUGGIE_LAUNCH_VERBOSE` | `0` | Enable verbose diagnostic logging |

---

## CLI Options

```bash
auggie-launch [launcher options] -- [auggie args]

Launcher options:
  --check, --9router-doctor   Test connectivity, 9router health, and models
  --models                    List all models discovered from 9router
  --combos                    List 9router combos and fallback groups
  --stats, --usage            Show 9router token savings and provider status
  --start-9router             Start 9router (installs it via npm if missing)
  --install-9router           Install 9router globally (npm i -g 9router@latest)
  --update-9router            Update 9router to the latest npm release
  --restore-9router-db        Restore ~/.9router/db.json from the bundled backup
  --print-env                 Show resolved config
  --proxy-only                Run only the local proxy in foreground
  --help, -h                  Show this help
```

---

## Testing

Run the included offline test suite:

```bash
python3 -m unittest -v test_modern_proxy.py
```
