---
name: ship
description: Verify, commit and push without leaking secrets
---

Ship the current work: gate it, scan for secrets, commit and push.

1. Run the full gate (as in /check). Do not continue while anything is red.
2. Scan for anything that must not be committed:

   ```bash
   git status --short | grep -iE '\.env[^.]|\.augment|\.bak' || echo CLEAN
   ```

   If that matches, stop and fix `.gitignore`; do not commit it.
3. Commit with a message explaining *why*, not just what.
4. Push to `origin main` and report the resulting commit range.

Always state what was **not** verified. If a change could not be exercised end to
end, say so instead of implying it was tested.
