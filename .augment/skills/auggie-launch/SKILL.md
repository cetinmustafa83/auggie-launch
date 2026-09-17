---
name: auggie-launch proxy
description: Architecture and failure modes of the auggie-launch local proxy
type: agent_requested
---

# auggie-launch proxy

`auggie-launch` is a local HTTP proxy that makes the Auggie CLI talk to CodeGPT
Plus instead of Augment's own backend.

## Request path

```
auggie -> local proxy (/chat-stream) -> CodeGPT bridge (/chat/tools/codegpt)
```

The proxy rewrites Auggie's request into the bridge shape: `modelId` (not
`model`), a `session_id`, an `X-Provider` header naming the model's upstream, and
tool messages folded to plain text.

## Failure modes to check first

| Symptom | Likely cause |
|---|---|
| `Tool X not found` | Model invented a tool name; add it to `_TOOL_ALIASES` |
| Command returns no output | `launch-process` missing `keep_stdin_open` |
| Answer stalls mid-task | `AUGGIE_LAUNCH_STREAM_THINKING` leaking think tags into history |
| History compacted too early | Context read as 200k instead of the real window |
| `grep` finds nothing for a glob | Filename globs must go to `find`, not `grep` |

## Diagnostics

```bash
auggie-launch --check      # connectivity, token, models
auggie-launch --models     # resolved model list
auggie-launch --sessions   # saved sessions with timestamps
```

`AUGGIE_LAUNCH_VERBOSE=1` shows per-request decisions: tool remaps, payload
sizes, upstream attempts.
