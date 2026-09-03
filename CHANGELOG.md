# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

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
- `main.py` renamed to `auggie_launch.py` (importable, installable module).
- Single source of truth for the version (`auggie_launch.__version__`).

### Fixed
- Ctrl+C no longer hangs on `httpd.shutdown()`.
- `load_config()` refreshes the module-level 9router state instead of shadowing it.
- Duplicate catalog error log removed; corrupted README section rebuilt.

## [0.3.0] - 2026-09-02

### Added
- Deep 9router local integration: combos, aliases, provider keys, catalog context
  windows, Caveman mode, MCP tool generation, and full Auggie CLI injection.
