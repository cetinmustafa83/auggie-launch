---
name: debug-launch
description: Diagnose auggie-launch when a request fails or stalls
---

Diagnose methodically; each step must produce evidence.

1. `auggie-launch --check` — upstream reachability, token, model list.
2. `auggie-launch --print-env` — active base URL, model, provider, context window.
3. `AUGGIE_LAUNCH_VERBOSE=1 auggie-launch --print "Reply with exactly: PING"` —
   look for `tool remap:`, `stream payload:` and `upstream attempt=`.
   A payload that keeps growing means history is not compacted; a repeated remap
   means the model keeps inventing the same unknown tool.

Report the evidence per step, then name the single most likely cause. Never
present a guess as a finding.
