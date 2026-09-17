---
name: Python quality gate
description: Run ruff, mypy and the unit tests after any Python change
type: always_apply
---

# Python quality gate

Every time you change a Python file in this workspace, finish the turn by running
the project's checks and reporting the result. Do not claim a change is done
until all three pass.

```bash
ruff check <changed paths>
mypy <changed paths>
python3 -m unittest test_modern_proxy
```

- A red `mypy` is a real failure, not noise. Fix the types.
- Never widen a type to `Any` purely to silence a checker.
- Report command output verbatim; do not paraphrase a pass.
