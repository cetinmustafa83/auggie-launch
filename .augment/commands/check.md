---
name: check
description: Run the full quality gate (ruff, mypy, tests)
---

Run the project's complete check suite and report each result separately.
Never summarise as "passed" without showing the command output.

```bash
cd /Users/fatmacetin/Desktop/auggie-launch
python3 -m ruff check auggie_launch/ test_modern_proxy.py
python3 -m mypy auggie_launch/
python3 -m unittest test_modern_proxy
```

Healthy output looks like: `All checks passed!` / `Success: no issues found in N
source files` / `Ran N tests` then `OK`.

A red `mypy` is a real failure: CI runs it, so leaving it red breaks the build.
