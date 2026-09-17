from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

from . import config

# ============================================================================
# End-to-end diagnostics
# ============================================================================
#
# Debugging this proxy by hand meant checking the same handful of things every
# time: is the token alive, does the catalog resolve, did the MCP servers start,
# is the model's provider correct. Each check below prints evidence, so a
# failing run points at one cause instead of a list of guesses.

PASS = "OK"
WARN = "WARN"
FAIL = "FAIL"


def _report(status: str, label: str, detail: str = "") -> None:
    marker = {PASS: "  [OK]  ", WARN: "  [WARN]", FAIL: "  [FAIL]"}[status]
    line = f"{marker} {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def check_python() -> str:
    import sys

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 10)
    _report(PASS if ok else FAIL, "Python", version + ("" if ok else " (3.10+ required)"))
    return PASS if ok else FAIL


def installed_auggie_version() -> str:
    """The CLI's own version string, or "" when it cannot be read."""
    binary = shutil.which(config.AUGGIE_BIN)
    if not binary:
        return ""
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=20)
    except Exception:
        return ""
    text = (proc.stdout or proc.stderr).strip()
    return text.split()[0] if text else ""


def latest_auggie_version() -> str:
    """The newest published version, or "" when offline or npm is missing."""
    if not shutil.which("npm"):
        return ""
    try:
        proc = subprocess.run(
            ["npm", "view", config.AUGGIE_PACKAGE, "version"],
            capture_output=True, text=True, timeout=45,
        )
    except Exception:
        return ""
    return (proc.stdout or "").strip().splitlines()[-1].strip() if proc.stdout else ""


def check_auggie_binary(check_updates: bool = True) -> str:
    path = shutil.which(config.AUGGIE_BIN)
    if not path:
        _report(FAIL, "auggie binary", f"'{config.AUGGIE_BIN}' not on PATH")
        return FAIL
    installed = installed_auggie_version()
    _report(PASS, "auggie binary", f"{path}" + (f" (v{installed})" if installed else ""))

    if not check_updates or not installed:
        return PASS
    latest = latest_auggie_version()
    if not latest:
        _report(WARN, "auggie version", f"v{installed}; could not reach npm to compare")
        return WARN
    if latest == installed:
        _report(PASS, "auggie version", f"v{installed} (latest)")
        return PASS
    _report(WARN, "auggie version", f"v{installed} installed, v{latest} available — run: auggie-launch --update-auggie")
    return WARN


def update_auggie() -> int:
    """Installs the latest CLI globally. Returns a process exit code."""
    if not shutil.which("npm"):
        print("error: npm is required to update the CLI", file=sys.stderr)
        return 1
    print(f"installing {config.AUGGIE_PACKAGE}@latest ...")
    command = ["npm", "install", "-g", f"{config.AUGGIE_PACKAGE}@latest"]
    if os.geteuid() != 0:
        # Same rule as install.sh: a non-root user installs into ~/.local.
        command.extend(["--prefix", os.path.join(os.path.expanduser("~"), ".local")])
    result = subprocess.run(command)
    if result.returncode != 0:
        print("error: update failed", file=sys.stderr)
        return result.returncode
    print(f"updated to v{installed_auggie_version() or 'unknown'}")
    return 0


def check_codegpt_token() -> tuple[str, str]:
    """Returns (status, token). The token is never printed."""
    if not config.IS_CODEGPT:
        _report(PASS, "Upstream", f"{config.TARGET_BASE_URL} (not CodeGPT)")
        return PASS, ""
    if not config.CODEGPT_TOKEN:
        _report(FAIL, "CodeGPT token", "not set and no sidecar to read it from")
        return FAIL, ""
    token = config.CODEGPT_TOKEN
    source = "pinned in env"
    if not token:
        from . import codegpt

        token = codegpt.session_token()
        source = "read from sidecar"
    if not token:
        _report(FAIL, "CodeGPT token", "unavailable")
        return FAIL, ""
    _report(PASS, "CodeGPT token", f"{len(token)} chars, {source}")
    return PASS, token


def check_catalog() -> str:
    if not config.IS_CODEGPT:
        return PASS
    from . import codegpt

    paths = codegpt._catalog_paths()
    models = codegpt.load_catalog_models()
    live = bool(paths) and os.path.exists(paths[0])
    detail = f"{len(models)} inclusive model(s)"
    if live:
        _report(PASS, "Model catalog", f"{detail}, read from {os.path.basename(os.path.dirname(os.path.dirname(paths[0])))}")
    else:
        _report(WARN, "Model catalog", f"{detail}, extension not installed — using the built-in list")
    return PASS if live else WARN


def check_model_resolution() -> str:
    from . import codegpt
    from .models import effective_context_limit

    model = config.TARGET_MODEL
    window = effective_context_limit(model)
    provider = codegpt.provider_for_model(model) if config.IS_CODEGPT else ""
    known = {entry["id"] for entry in codegpt.load_catalog_models()} if config.IS_CODEGPT else set()

    detail = f"{model}"
    if window:
        detail += f", context {window:,}"
    if provider:
        detail += f", provider {provider}"
    if known and model not in known:
        _report(WARN, "Model", detail + " (not in the inclusive list)")
        return WARN
    _report(PASS, "Model", detail)

    if config.CODEGPT_PROVIDER and provider != config.CODEGPT_PROVIDER:
        _report(WARN, "Provider override", f"{config.CODEGPT_PROVIDER} pinned but the catalog says {provider}")
        return WARN
    return PASS


def check_upstream_connection() -> str:
    parsed = urllib.parse.urlparse(config.TARGET_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=5.0):
            pass
    except OSError as exc:
        _report(FAIL, "Upstream socket", f"{host}:{port} unreachable — {exc}")
        return FAIL
    _report(PASS, "Upstream socket", f"{host}:{port}")
    return PASS


def check_codegpt_roundtrip(token: str) -> str:
    """A real request, so a bad token or provider shows up here and not later."""
    if not config.IS_CODEGPT or not token:
        return PASS
    from . import codegpt

    url = upstream_url_for_doctor()
    headers = codegpt.extra_headers()
    headers["Content-Type"] = "application/json"
    body = codegpt.adapt_request_body({
        "model": config.TARGET_MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    })
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read(256)
        _report(PASS, "Upstream round trip", "answered")
        return PASS
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:160]
        _report(FAIL, "Upstream round trip", f"HTTP {exc.code} — {detail}")
        return FAIL
    except Exception as exc:
        _report(FAIL, "Upstream round trip", str(exc)[:160])
        return FAIL


def upstream_url_for_doctor() -> str:
    from .upstream import upstream_url

    return upstream_url()


def check_mcp() -> str:
    from .injections import generate_injected_mcp_config

    path = generate_injected_mcp_config()
    if not path or not os.path.isfile(path):
        _report(WARN, "MCP config", "not generated")
        return WARN
    with open(path, encoding="utf-8") as fh:
        servers = (json.load(fh) or {}).get("mcpServers") or {}
    if not servers:
        _report(WARN, "MCP servers", "none configured")
        return WARN
    problems: list[str] = []
    for name, spec in servers.items():
        command = str((spec or {}).get("command") or "")
        if command == "npx":
            # npx re-resolves on every launch and can exceed Auggie's 10s window.
            problems.append(f"{name} uses npx")
            continue
        if command and not (os.path.isfile(command) or shutil.which(command)):
            problems.append(f"{name} command missing")
    if problems:
        _report(WARN, "MCP servers", f"{len(servers)} configured; " + ", ".join(problems))
        return WARN
    _report(PASS, "MCP servers", ", ".join(sorted(servers)))
    return PASS


def check_tool_mapping() -> str:
    """Confirms the alias table still covers the names models actually emit."""
    from .codegpt import resolve_tool_name, route_tool_call

    available = {
        "view", "save-file", "str-replace-editor", "launch-process",
        "remove-files", "web-fetch",
    }
    invented = {
        "read": "view",
        "read_file": "view",
        "bash": "launch-process",
        "execute_terminal_command": "launch-process",
        "write_file": "save-file",
        "edit_file": "str-replace-editor",
        "delete_file": "remove-files",
        "fetch": "web-fetch",
    }
    broken = [name for name, expected in invented.items() if resolve_tool_name(name, available) != expected]
    if broken:
        _report(FAIL, "Tool name mapping", "unresolved: " + ", ".join(broken))
        return FAIL

    search, _ = route_tool_call("glob_search", {"pattern": "x"}, available | {"launch-process"})
    if search not in available:
        _report(FAIL, "Search routing", f"resolved to '{search}', which is not a servable tool")
        return FAIL
    _report(PASS, "Tool name mapping", f"{len(invented)} aliases, search shaped correctly")
    return PASS


def check_repo_hygiene() -> str:
    """Guards the two ways this repo leaks: tracked secrets and untracked rules."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems: list[str] = []
    env = os.path.join(root, ".env")
    if os.path.isfile(env):
        mode = os.stat(env).st_mode & 0o777
        if mode & 0o077:
            problems.append(f".env mode {mode:o} is world/group readable")
    rules_dir = os.path.join(root, ".augment", "rules")
    if os.path.isdir(rules_dir):
        rules = [f for f in os.listdir(rules_dir) if f.endswith(".md")]
        if not rules:
            problems.append("no rules in .augment/rules")
    else:
        problems.append(".augment/rules missing")
    if problems:
        _report(WARN, "Repo hygiene", "; ".join(problems))
        return WARN
    _report(PASS, "Repo hygiene", ".env private, project rules present")
    return PASS


def check_local_state() -> str:
    """`prompt-history.jsonl` records every prompt in plain text."""
    path = os.path.join(os.path.expanduser("~"), ".augment", "prompt-history.jsonl")
    if not os.path.isfile(path):
        return PASS
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        _report(WARN, "Prompt history", f"mode {mode:o} — readable by others; chmod 600 it")
        return WARN
    _report(PASS, "Prompt history", "mode 600")
    return PASS


def run_doctor(check_updates: bool = True) -> int:
    """Runs every check. Returns a process exit code."""
    print("=" * 62)
    print("auggie-launch doctor")
    print("=" * 62)
    print(f"  upstream : {config.TARGET_BASE_URL}")
    print(f"  model    : {config.TARGET_MODEL}")
    print(f"  mode     : {'CodeGPT Plus' if config.IS_CODEGPT else 'OpenAI-compatible'}")
    print("-" * 62)

    results: list[str] = []
    results.append(check_python())
    results.append(check_auggie_binary(check_updates=check_updates))
    token_status, token = check_codegpt_token()
    results.append(token_status)
    results.append(check_catalog())
    results.append(check_model_resolution())
    results.append(check_upstream_connection())
    results.append(check_codegpt_roundtrip(token))
    results.append(check_mcp())
    results.append(check_tool_mapping())
    results.append(check_repo_hygiene())
    results.append(check_local_state())

    print("-" * 62)
    failures = results.count(FAIL)
    warnings = results.count(WARN)
    if failures:
        print(f"{failures} failure(s), {warnings} warning(s)")
        return 1
    print("all checks passed" + (f" ({warnings} warning(s))" if warnings else ""))
    return 0
