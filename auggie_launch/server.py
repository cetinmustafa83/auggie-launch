from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from typing import Any

from . import config
from .models import fetch_upstream_models, model_context_limit

# ============================================================================
# Local proxy server lifecycle & introspection
# ============================================================================


# Tools that change the workspace. A planning turn must not run any of them.
_MUTATING_TOOLS = (
    "str-replace-editor",
    "save-file",
    "remove-files",
    "launch-process",
    "git-commit",
)


def permissions_for_mode(mode: str) -> list[str] | None:
    """Maps a session mode onto `--permission` arguments.

    The CLI takes one `tool:policy` pair per flag and defaults to asking, so a
    mode is expressed by naming the tools it should not have to ask about:

      plan         read-only -- every mutating tool is denied
      code         writes allowed, but the shell still asks
      full-access  nothing asks, shell included

    Privilege is not escalated by the proxy. In full-access the shell is simply
    unrestricted, so a `sudo` the model decides it needs is run by the CLI and
    the terminal prompts for the password. The proxy never sees or stores it.

    Returns None for an unknown mode so the caller can report it.
    """
    normalised = (mode or "").strip().lower().replace("_", "-")
    if normalised == "plan":
        return [f"{tool}:deny" for tool in _MUTATING_TOOLS]
    if normalised == "code":
        return [f"{tool}:allow" for tool in _MUTATING_TOOLS if tool != "launch-process"]
    if normalised in {"full-access", "fullaccess", "yolo"}:
        return [f"{tool}:allow" for tool in _MUTATING_TOOLS]
    return None


def mode_runs_until_done(mode: str) -> bool:
    """True when the mode should not stop at a turn limit.

    `--print` is bounded by `--max-turns`; the interactive modes are already
    unbounded. full-access means "finish the task", so it lifts the ceiling.
    """
    return (mode or "").strip().lower().replace("_", "-") in {"full-access", "fullaccess", "yolo"}


_SECRET_MARKERS = ("TOKEN", "API_KEY", "SECRET", "PASSWORD", "SIGNED_DISTINCT")

# Ceiling for one post-run check; enoug h for a slow suite, short enough to notice.
CHECK_TIMEOUT_SECONDS = int(os.environ.get("AUGGIE_LAUNCH_CHECK_TIMEOUT") or "300")


def _is_secret_key(name: str) -> bool:
    """True for environment names whose value must not reach a child process."""
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def post_run_checks() -> list[dict[str, Any]]:
    """Runs the project's quality gate and returns one row per check.

    Mirrors `make check`: a task is only done when these are green, so the
    launcher reports them instead of leaving the verdict to the model.
    """

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checks = [
        ("lint", [sys.executable, "-m", "ruff", "check", "auggie_launch/", "test_modern_proxy.py", "test_install.py"]),
        ("types", [sys.executable, "-m", "mypy", "auggie_launch/"]),
        ("tests", [sys.executable, "-m", "unittest", "test_modern_proxy", "test_install"]),
    ]
    # A child inherits this process's environment, which carries the live token.
    # Keep it: a test that dumps os.environ would print it. Nothing the checks
    # need is secret, so the sensitive keys are dropped first.
    child_env = {k: v for k, v in os.environ.items() if not _is_secret_key(k)}

    results: list[dict[str, Any]] = []
    for name, command in checks:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, cwd=root,
                                  timeout=CHECK_TIMEOUT_SECONDS, env=child_env)
            output = (proc.stdout + proc.stderr).strip()
            results.append({
                "name": name,
                "ok": proc.returncode == 0,
                "command": " ".join(shlex.quote(part) for part in command),
                "output": output[-4000:],
            })
        except subprocess.TimeoutExpired:
            # Distinguish a timeout from a real failure: "check failed" would send
            # someone looking for a broken assertion that does not exist.
            results.append({
                "name": name,
                "ok": False,
                "command": " ".join(shlex.quote(part) for part in command),
                "output": f"timed out after {CHECK_TIMEOUT_SECONDS}s -- raise "
                          f"AUGGIE_LAUNCH_CHECK_TIMEOUT if the suite is legitimately slow",
            })
        except Exception as exc:
            results.append({
                "name": name,
                "ok": False,
                "command": " ".join(shlex.quote(part) for part in command),
                "output": f"{type(exc).__name__}: {exc}",
            })
    return results


def write_failure_todo(results: list[dict[str, Any]], path: str = "") -> str:
    """Appends failing checks to a TODO file and returns its path.

    A red gate should leave a durable record rather than scrolling past in the
    terminal, so the next session (or the next person) picks it up.
    """
    failures = [row for row in results if not row.get("ok")]
    if not failures:
        return ""
    target = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TODO.md"
    )
    stamp = time.strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    if not os.path.isfile(target):
        lines.append("# TODO")
        lines.append("")
        lines.append("Written by auggie-launch when the post-run gate fails.")
        lines.append("")
    lines.append(f"## {stamp} — post-run checks failed")
    lines.append("")
    for row in failures:
        lines.append(f"- [ ] **{row['name']}** — `{row.get('command', '')}`")
        tail = [ln for ln in str(row.get("output") or "").splitlines() if ln.strip()][-6:]
        for line in tail:
            lines.append(f"      {line}")
    lines.append("")
    with open(target, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return target


def report_post_run_checks(results: list[dict[str, Any]]) -> bool:
    """Prints the gate and returns True only when everything passed."""
    if not results:
        return True
    print()
    print("=" * 62)
    print("post-run checks")
    print("=" * 62)
    for row in results:
        mark = "PASS" if row["ok"] else "FAIL"
        print(f"  [{mark}] {row['name']}")
        if not row["ok"]:
            for line in str(row.get("output") or "").splitlines()[-12:]:
                print(f"         {line}")
    failures = [row for row in results if not row["ok"]]
    print("-" * 62)
    if failures:
        print(f"{len(failures)} check(s) FAILED -- fix before treating the task as done")
        try:
            target = write_failure_todo(results)
            if target:
                print(f"recorded in {target}")
        except Exception as exc:
            print(f"could not record the failures: {exc}")
        return False
    print("all checks green")
    return True


def find_free_port() -> int:
    """Binds an ephemeral loopback port and reports it, for `PORT=0`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_server(httpd: ThreadingHTTPServer) -> None:
    """Shuts the proxy down without ever blocking the exit path (e.g. on Ctrl+C).

    Interrupts that arrive while we are already tearing down (a user impatiently
    holding Ctrl+C) are swallowed so the process can still exit cleanly.
    """
    try:
        stopper = threading.Thread(target=httpd.shutdown, daemon=True)
        stopper.start()
        stopper.join(timeout=2.0)
    except KeyboardInterrupt:
        pass
    try:
        httpd.server_close()
    except Exception:
        pass


def print_models() -> None:
    """Lists the models this proxy will advertise to the CLI."""

    seen: set[str] = set()
    print("Configured model:")
    print(f"  - {config.TARGET_MODEL}")
    seen.add(config.TARGET_MODEL)

    if config.IS_CODEGPT:
        from . import codegpt as _cg  # local: avoid a cycle at import time
        for label, tier in (("Included with the subscription", "unlimited"),
                            ("Metered (draws on the credit allowance)", "metered")):
            rows = _cg.models_by_tier(tier)
            if not rows:
                continue
            print(f"\n{label}:")
            for entry in rows:
                marker = " (active)" if entry["id"] == config.TARGET_MODEL else ""
                panel = entry.get("panel_name") or entry["id"]
                alias = f"  (panel: {panel})" if panel != entry["id"] else ""
                context = f"  {entry['context']:,} ctx" if entry.get("context") else ""
                print(f"  - {entry['id']}  [{entry['provider']}]{context}{alias}{marker}")
                seen.add(entry["id"])
        print("\nOnly one session at a time: the subscription is per-user, so a"
              "\nsecond concurrent auggie-launch will contend with this one.")
        return

    live = fetch_upstream_models()
    if live:
        print("\nModels reported by the upstream:")
        for item in live:
            mid = item.get("id") or item.get("name")
            if mid and mid not in seen:
                print(f"  - {mid}")
                seen.add(mid)


def print_env(port: int) -> None:
    """Prints the resolved configuration, secrets reduced to presence only."""
    from . import codegpt

    print(f"AUGGIE_BIN={config.AUGGIE_BIN}")
    print(f"AUGGIE_LAUNCH_PROXY_URL=http://127.0.0.1:{port}")
    print(f"AUGGIE_LAUNCH_BASE_URL={config.TARGET_BASE_URL}")
    print(f"AUGGIE_LAUNCH_MODEL={config.TARGET_MODEL}")
    print(f"AUGGIE_LAUNCH_API_KEY={'<set>' if config.API_KEYS else ''}")
    mode = "codegpt" if config.IS_CODEGPT else "openai-compatible"
    print(f"AUGGIE_LAUNCH_UPSTREAM_MODE={mode}")
    if config.IS_CODEGPT:
        provider = codegpt.provider_for_model(config.TARGET_MODEL) or "(unresolved)"
        print(f"AUGGIE_LAUNCH_CODEGPT_HARNESS={config.CODEGPT_HARNESS}")
        print(f"AUGGIE_LAUNCH_CODEGPT_PROVIDER={provider}")
        print(f"AUGGIE_LAUNCH_CODEGPT_TOKEN={'<set>' if codegpt.session_token() else '(missing)'}")
        print(f"AUGGIE_LAUNCH_CODEGPT_MODELS={','.join(m['id'] for m in codegpt.load_catalog_models())}")
    print(f"AUGGIE_LAUNCH_INDEXING_MODE={config.INDEXING_MODE}")
    print(f"AUGGIE_LAUNCH_STREAM_THINKING={'true' if config.STREAM_THINKING else 'false'}")
    print(f"AUGGIE_LAUNCH_USER_AGENT={config.UPSTREAM_USER_AGENT}")
    print(f"AUGGIE_LAUNCH_UPSTREAM_APP_NAME={config.UPSTREAM_APP_NAME}")
    print(f"AUGGIE_LAUNCH_MODEL_CONTEXT={model_context_limit(config.TARGET_MODEL)}")
    print("loaded_env_files=" + (", ".join(config._LOADED_ENV_FILES) if config._LOADED_ENV_FILES else "(none)"))
