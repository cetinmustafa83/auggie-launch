# auggie-launch — working notes

A local HTTP proxy that lets the Auggie CLI talk to CodeGPT Plus instead of
Augment's own backend. Python, no runtime dependencies beyond the standard
library.

## Layout

```
auggie_launch/
  cli.py         entry point, launcher flags
  proxy.py       HTTP handler, streaming, tool-call remapping
  codegpt.py     CodeGPT bridge: token, headers, body rewriting, tool routing
  upstream.py    connection pool, retries, backoff
  transform.py   Augment → OpenAI request/message/tool conversion
  truncation.py  token accounting, turn-atomic truncation, tool-call merge
  registry.py    model list and feature flags served to Auggie
  config.py      env resolution, everything is configurable
```

## Commands

```bash
python3 -m ruff check auggie_launch/ test_modern_proxy.py
python3 -m mypy auggie_launch/
python3 -m unittest test_modern_proxy
auggie-launch --check            # connectivity + token + models
auggie-launch --models           # resolved model list
auggie-launch --sessions         # saved sessions, with timestamps
```

## Invariants worth protecting

- **Tests and `mypy` must stay green.** CI runs both; a red `mypy` breaks the
  build even though the code still runs.
- **Secrets never enter the repo.** `.env`, `.env.bak*` and `.augment/` are
  ignored; keep it that way.
- **Tool names are matched by shape**, not by a fixed list, so unseen spellings
  from new models still resolve.
- **Stream deltas carry only `delta`.** Sending the same text under both `text`
  and `delta` makes Auggie render it twice.
- **`launch-process` needs `keep_stdin_open`**, otherwise a command that exits 0
  still returns empty output and the model retries forever.

## How to work here

Prefer measuring over reasoning about behaviour: run the command, read the
output, then conclude. Several bugs in this codebase were only visible in a real
request (`grep` returning nothing for a glob, output swallowed by a missing
flag). When something is unverified, say so rather than implying it works.
