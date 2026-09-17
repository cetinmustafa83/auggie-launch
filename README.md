# auggie-launch

High-performance, modern launcher for the installed `auggie` / Augment Code CLI that starts a local
**Python** Augment-compatible proxy and forwards chat traffic to standard OpenAI-compatible endpoints,

**No secrets or private endpoints are hard-coded.** Configure upstream model
settings through `.env` or environment variables.



---

# auggie-launch

Run the [Auggie CLI](https://www.npmjs.com/package/@augmentcode/auggie) against
**CodeGPT Plus** instead of Augment's own backend.

`auggie-launch` starts a small local HTTP proxy on loopback, points the CLI at
it, and translates each request into the shape CodeGPT's bridge expects. Python
3.10+, standard library only.

```bash
./install.sh                 # installs into ~/.local/bin
auggie-launch                # interactive session
auggie-launch --print "hi"   # one-shot
```

## Why a proxy

CodeGPT Plus does not expose a plain OpenAI `/chat/completions`. The CLI calls a
different path, authenticates differently, and expects a different request
shape. Rather than patch the CLI, the launcher stands in front of it:

```
auggie → 127.0.0.1 (local proxy) → api.codegpt.co/api/v1/chat/tools/<harness>
```

The proxy:

- reads the session token from the CodeGPT VS Code extension, or uses one you pin;
- addresses the model as `modelId` and names its upstream in an `X-Provider`
  header (resolved per model from the extension's own catalog);
- folds tool traffic into plain text, which is the only form the agent backend
  accepts;
- remaps tool names the model invents (`file_search`, `read_file`, `bash`, ...)
  onto the ones the CLI actually has.

## Models

The inclusive ("economy") tier is read live from the CodeGPT extension's
`model-catalog.json`, so the list follows your plan instead of being pinned in
this repo. `--models` shows what will be advertised:

```
$ auggie-launch --models
Configured model:
  - deepseek-v4.1-flash

Inclusive (economy) models:
  - deepseek-v4.1-flash  [openrouter] (active)
  - deepseek-v4-flash    [openrouter]
  - gemini-3.8-flash     [vertexai]
  ...
```

Switching model needs no other change — the provider is derived from the
catalog:

```bash
AUGGIE_LAUNCH_MODEL=gemini-3.8-flash
```

## Configuration

Everything lives in `~/.config/auggie-launch/.env` (or `./.env`), template in
`.env.example`. The minimum for CodeGPT Plus:

```bash
AUGGIE_LAUNCH_BASE_URL=https://api.codegpt.co/api/v1
AUGGIE_LAUNCH_MODEL=deepseek-v4.1-flash
AUGGIE_LAUNCH_CODEGPT_TOKEN=session-...
```

With a pinned token the extension does not need to be running. Without one, the
launcher reads the live session from `http://localhost:54112/api/session`.

Any other OpenAI-compatible upstream works too — point
`AUGGIE_LAUNCH_BASE_URL` at it and set `AUGGIE_LAUNCH_API_KEY`.

## Diagnostics

```bash
auggie-launch --doctor     # end-to-end: token, catalog, upstream, MCP, tools
make check                 # lint + types + tests
```

`--doctor` prints evidence per check, so a failure names one cause rather than
leaving you to guess:

```
  [OK]  CodeGPT token — 57 chars, pinned in env
  [OK]  Model — deepseek-v4.1-flash, context 1,048,576, provider openrouter
  [WARN] Repo hygiene — .env mode 664 is world/group readable
```

## Sessions

```bash
auggie-launch --sessions           # saved sessions, with timestamps
auggie-launch --resume <id>        # resume one (picker when the id is omitted)
auggie-launch -c                   # resume the most recent
```

Inside an interactive session, `/sessions` opens the CLI's own picker.

## Rules, Commands, Skills and Memory

Auggie reads project guidance from the workspace. `auggie-launch` does not
intercept any of it, so these work exactly as they do against Augment.

### Rules (`always_apply` is the important one)

`.augment/rules/*.md` and `~/.augment/rules/*.md` are loaded automatically. The
`type` frontmatter decides when a rule applies:

| `type` | When it runs |
|---|---|
| `always_apply` | Every turn, automatically |
| `agent_requested` | When the model decides it is relevant |
| `manual` | Only when invoked |

```markdown
---
name: Python quality gate
description: Run ruff, mypy and the tests after any Python change
type: always_apply
---

<rule body>
```

Auggie also reads, in this order of precedence: `AGENTS.md`, `CLAUDE.md`,
`.augment-guidelines`, plus Cursor (`.cursor/rules/`, `.cursorrules`) and
Windsurf (`.windsurf/rules/`, `.windsurfrules`) equivalents.

### Commands

`.augment/commands/*.md` become `/name` in the CLI. This repo ships three:
`/check`, `/ship`, `/debug-launch`.

### Skills

`.augment/skills/<name>/SKILL.md` — the filename must be exactly `SKILL.md`, and
`name` plus `description` are required in the frontmatter, or the skill is
rejected.

```markdown
---
name: auggie-launch proxy
description: Architecture and failure modes of the local proxy
type: agent_requested
---
```

### Memory

Auggie persists conversations under `~/.augment/`:

| Path | Contents |
|---|---|
| `sessions/*.json` | Full transcripts; `auggie-launch --sessions` lists them |
| `prompt-history.jsonl` | Every prompt you have sent |
| `task-storage/` | Task state |

**`prompt-history.jsonl` is stored in plain text with mode 644.** Never paste a
credential into a prompt, and keep the file `chmod 600`. Use `/resume`
(`auggie-launch --resume`) to continue a past conversation rather than repeating
its context.

### Deliberately disabled

`enable_hindsight` stays off. It is Augment's cloud code-index engine: enabling
it would upload this workspace to Augment's servers, contradicting
`AUGGIE_LAUNCH_INDEXING_MODE=complete`, and the index would be useless anyway
because requests go to CodeGPT rather than Augment.

## CLI Options

```bash
auggie-launch [launcher options] -- [auggie args]

Launcher options:
  --check, --doctor           End-to-end diagnostics (token, catalog, MCP, tools)
  --models                    List the models served to the CLI
  --print-env                 Show resolved config
  -c, --continue              Resume the most recent session
  --resume [sessionId]        Resume a session (picker when the id is omitted)
  --sessions                  List saved sessions for this workspace
  --proxy-only                Run only the local proxy in foreground
  --help, -h                  Show this help
```

---

## Security

The proxy binds to `127.0.0.1` only and mints a **random per-session token** that it
injects into the Auggie process. Every request except `/health` and the token endpoint
must present it (`Authorization: Bearer …`), so no other local process can spend your
upstream credits. Pin it with `AUGGIE_LAUNCH_LOCAL_TOKEN`, or disable the check with
`AUGGIE_LAUNCH_REQUIRE_LOCAL_TOKEN=false`.

---

## Development

The launcher is a package with one module per concern:

| Module | Responsibility |
|---|---|
| `auggie_launch/config.py` | Shared runtime state, `.env` loading, `load_config()` |
| `auggie_launch/truncation.py` | Token estimation, turn-atomic truncation, tool-call merging |
| `auggie_launch/transform.py` | Augment ↔ OpenAI message/tool transformation |
| `auggie_launch/upstream.py` | Connection pool, retries, throttling, tunnel failover |
| `auggie_launch/models.py` | Catalog lookups, per-model context windows, model routing |
| `auggie_launch/registry.py` | Model registry and session payloads served to Auggie |
| `auggie_launch/proxy.py` | The local Augment-compatible HTTP server |
| `auggie_launch/injections.py` | Environment, feature-flag, and MCP injections |
| `auggie_launch/doctor.py` | end-to-end diagnostics |
| `auggie_launch/cli.py` | Argument handling and process launch |

```bash
python3 -m unittest -v test_modern_proxy   # offline test suite
python3 -m ruff check .                    # lint
python3 -m mypy                            # type check
python3 -m build                           # build wheel/sdist
```

CI runs the same steps on Python 3.10–3.13 (`.github/workflows/ci.yml`).
See `CHANGELOG.md` for release notes and `LICENSE` for terms (research/learning only).
