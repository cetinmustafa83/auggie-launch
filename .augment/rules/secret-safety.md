---
name: Secret safety
description: Keep credentials out of commits, tool output and chat transcripts
type: always_apply
---

# Secret safety

This workspace holds a live provider token, so credentials leak easily.

Never print, paste or echo the value of `AUGGIE_LAUNCH_CODEGPT_TOKEN`,
`AUGGIE_LAUNCH_CODEGPT_SIGNED_DISTINCT_ID`, `AUGGIE_LAUNCH_TAVILY_API_KEY`,
`AUGGIE_LAUNCH_API_KEY` or `AUGGIE_LAUNCH_API_KEYS`, and never put one in a
command you run.

Before any commit:

```bash
git status --short | grep -iE '\.env[^.]|\.augment|\.bak' || echo CLEAN
```

If a secret is exposed, say so and recommend rotating it at the source.
Rotating the credential is the only real fix; editing the file is not.
