# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- The inclusive-model list is read live from the CodeGPT extension's catalog
  (context window, tool and vision support included) instead of being pinned in
  this repo, and `X-Provider` is derived from each model automatically.
- CodeGPT Plus inclusive ("economy") models: the bridge is
  `POST /chat/tools/<harness>` addressed by `modelId`, with the model's upstream
  in an `X-Provider` header. No agent is involved, so deepseek-v4.1-flash and the
  other unlimited models are now reachable
  (`AUGGIE_LAUNCH_CODEGPT_PROVIDER`, `AUGGIE_LAUNCH_CODEGPT_HARNESS`).
- `AUGGIE_LAUNCH_REPLY_LANGUAGE` to pin replies to one language; upstreams
  otherwise drifted into an arbitrary language mid-answer.
- CodeGPT Plus upstream (`auggie_launch/codegpt.py`): auto-detected on
  `api.codegpt.co`, it mints a fresh session token from the CodeGPT VS Code
  extension, routes plain chats to `/chat/extension` and tool calls to
  `/chat/tools`, and rewrites OpenAI tools/`agentId` into the agent request shape.
- CodeGPT Plus models surfaced to Auggie from the extension's local DB
  (`codegpt_model_ids`), with a static plan fallback.
- `AUGGIE_LAUNCH_UPSTREAM_TIMEOUT` to configure the upstream request timeout
  (default `300` seconds) instead of the hard-coded value.

### Added
- Session compaction: the proxy advertises Auggie's history-summarization
  feature flags (`history_summary_min_version`, `history_summary_params`), so a
  long session is summarized instead of overflowing the upstream window
  (`AUGGIE_LAUNCH_HISTORY_SUMMARY*`).

### Fixed
- Reasoning is no longer streamed inside raw ` thinking...` text: the CLI renders
  reasoning itself, and the tags both leaked into the transcript and confused
  the model on later turns. `AUGGIE_LAUNCH_STREAM_THINKING` defaults to off.
- Streamed tool calls are remapped only after the fragments are merged.
  Remapping each delta on its own built a full command from an empty argument
  set and then concatenated the real arguments after it, producing invalid JSON
  and killing the stream.
- Timeouts: pooled HTTP connections kept the timeout of whichever request
  created them, so a short-lived call capped a later long one. The socket
  timeout is now refreshed on acquire.
- Timeouts: `npx -y <pkg>@latest` re-resolved on every launch (~25s), blowing
  past Auggie's 10s MCP window, so MCP servers failed on every run. An installed
  binary is used when present, with `npx --prefer-offline` as the fallback.
- Timeouts: `completion_timeout_ms` advertised to Auggie (600s) exceeded the
  proxy's own upstream cutoff (300s); it is now derived from that timeout.
- Search routing no longer targets `codebase-retrieval`: it is served by
  Auggie's `/agents` endpoint, which the proxy only stubs, so searches returned
  nothing and the model retried until it crashed. Searches now resolve to a real
  `grep`/`ls` via `launch-process` (patterns shell-quoted).
- Invented tool names (`file_search`, `read_file`, `bash`, ...) are remapped onto
  the real Auggie tool before the turn is replayed, instead of failing with
  "Tool X not found" and derailing the conversation.
- Tool messages are no longer passed to CodeGPT as OpenAI `tool` turns: Vertex
  rejects them with "number of function response parts is not equal to the
  number of function call parts", so tool results and assistant `tool_calls` are
  folded into plain user/assistant text.
- Parallel tool calls whose deltas all share `index: 0` (CodeGPT restarts the
  index per call) are kept separate instead of being concatenated into invalid
  JSON arguments.
- Streaming deltas no longer duplicate the assistant text: each chunk carries
  only `delta`, not the same text under both `text` and `delta`.
- CodeGPT/Vertex compatibility: conversations ending on an assistant turn are
  nudged with a user message (Vertex rejects "Requests ending with a model
  turn"), and integer `enum` members in tool schemas are coerced to strings.
- The `/model` picker no longer crashes: every registry entry now carries
  `displayName`/`shortName` as Auggie's menu requires.
- 9router combos/aliases are hidden when 9router is not the upstream.

### Changed
- Debug request dumps in the proxy are now written by a single
  `dump_debug_request` helper instead of two duplicated blocks.
- `/token`-suffixed paths no longer bypass local-token authorization.

## [0.4.0] - 2026-09-03

### Added
- Runtime Cloudflare-tunnel failover: chat and `/models` traffic switches to the
  9router tunnel when the local socket dies mid-session.
- Self-healing 9router install: `--install-9router`, `--update-9router`,
  `--restore-9router-db`, and automatic `npm i -g 9router@latest --prefer-online`
  when 9router is not detected (`AUGGIE_LAUNCH_AUTO_INSTALL_9ROUTER`).
- Bundled `~/.9router/db.json` restore from the repo's `9router/db/` backups.
- Local proxy authentication: per-session random token, verified on every request
  (`AUGGIE_LAUNCH_REQUIRE_LOCAL_TOKEN`).
- Packaging (`pyproject.toml`, `pipx`/`pip install .`, `auggie-launch` console
  script), ruff + mypy configuration, GitHub Actions CI, `LICENSE`, `CHANGELOG.md`.

### Changed
- Model context windows injected into Auggie are now per-model: combos use their
  weakest fallback member, aliases resolve through their target, and every model's
  `suggested_prefix/suffix_char_count` is derived from its real window.
- Requests honour the model Auggie asks for when it is one the launcher advertises,
  and cap `max_tokens`/`max_completion_tokens` to the model's budget.
- `main.py` split into the `auggie_launch` package (config, truncation, transform,
  upstream, models, registry, proxy, injections, ninerouter, cli).
- Single source of truth for the version (`auggie_launch.__version__`).

### Fixed
- Ctrl+C no longer hangs on `httpd.shutdown()`.
- `load_config()` refreshes the module-level 9router state instead of shadowing it.
- Duplicate catalog error log removed; corrupted README section rebuilt.

## [0.3.0] - 2026-09-02

### Added
- Deep 9router local integration: combos, aliases, provider keys, catalog context
  windows, Caveman mode, MCP tool generation, and full Auggie CLI injection.
