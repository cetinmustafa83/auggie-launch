from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

from . import config
from .config import load_config, log
from .doctor import run_doctor, update_auggie
from .injections import build_injected_environment, generate_injected_mcp_config
from .proxy import AuggieProxy
from .server import (
    acquire_session_lock,
    find_free_port,
    mode_runs_until_done,
    permissions_for_mode,
    print_env,
    print_models,
    release_session_lock,
    report_post_run_checks,
    stop_server,
)
from .upstream import upstream_url

# ============================================================================
# Main Entry Point
# ============================================================================

def print_sessions() -> None:
    """Lists saved sessions for the current workspace, most recent first.

    Mirrors the CLI's own picker (which needs a TTY) so the list is visible from
    a plain shell; `--resume <id>` accepts any unambiguous prefix of the id.
    """
    sessions_dir = os.path.join(os.path.expanduser("~"), ".augment", "sessions")
    if not os.path.isdir(sessions_dir):
        print("no saved sessions found")
        return

    workspace = os.getcwd()
    rows: list[tuple[float, str, str, int, str]] = []
    for entry in os.scandir(sessions_dir):
        if not entry.name.endswith(".json") or not entry.is_file():
            continue
        try:
            with open(entry.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        root = data.get("workspaceRoot") or data.get("workspace_root") or ""
        # The CLI filters by workspace; show the same view.
        if root and os.path.realpath(str(root)) != os.path.realpath(workspace):
            continue
        try:
            modified = os.path.getmtime(entry.path)
        except OSError:
            continue
        session_id = str(data.get("sessionId") or data.get("session_id") or entry.name[:-5])
        name = str(data.get("name") or data.get("title") or "(untitled)")
        history = data.get("chatHistory") or data.get("history") or []
        turns = len(history) if isinstance(history, list) else 0
        rows.append((modified, session_id, name, turns, root or workspace))

    rows.sort(key=lambda row: row[0], reverse=True)
    if not rows:
        print(f"no saved sessions for {workspace}")
        return

    print(f"Saved sessions for {workspace}\n")
    print(f"  {'#':>3}  {'Updated':16}  {'Turns':>5}  {'Session':38}  Name")
    for index, (modified, session_id, name, turns, _root) in enumerate(rows, start=1):
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(modified))
        print(f"  {index:>3}  {stamp:16}  {turns:>5}  {session_id[:38]:38}  {name[:32]}")
    print("\nResume with:  auggie-launch --resume <session id or prefix>")


def main() -> None:
    launcher_args = []
    pass_args = []
    args = sys.argv[1:]
    while args:
        arg = args.pop(0)
        if arg == "--":
            pass_args.extend(args)
            break
        # Session shortcuts: passed straight through, since the CLI builds its
        # own date-ordered, workspace-filtered picker for the bare forms.
        if arg in {"--continue", "-c"}:
            pass_args.append("--continue")
            continue
        if arg == "--resume":
            pass_args.append("--resume")
            if args and not args[0].startswith("-"):
                pass_args.append(args.pop(0))
            continue
        if arg in {"--sessions", "--list-sessions"}:
            launcher_args.append("--sessions")
            continue
        if arg == "--mode":
            if not args:
                print("error: --mode needs one of: plan, code, full-access", file=sys.stderr)
                sys.exit(2)
            launcher_args.extend(["--mode", args.pop(0)])
            continue
        if arg in {
            "--print-env",
            "--proxy-only",
            "--check",
            "--doctor",
            "--no-update-check",
            "--update-auggie",
            "--models",
            "--stats",
            "--reset-stats",
            "--sessions",
            "--mode",
            "--help",
            "-h",
        }:
            launcher_args.append(arg)
        else:
            pass_args.append(arg)

    if "--help" in launcher_args or "-h" in launcher_args:
        print("Usage: auggie-launch [launcher options] -- [auggie args]")
        print("       auggie-launch [auggie args]")
        print("\nLauncher options:")
        print("  --check, --doctor           End-to-end diagnostics (token, catalog, MCP, tools)")
        print("  --no-update-check           Skip the CLI version comparison during --doctor")
        print("  --update-auggie             Install the latest @augmentcode/auggie globally")
        print("  --models                    List the models served to the CLI")
        print("  --stats                     Show usage counters (requests, tools, remaps)")
        print("  --reset-stats               Clear the usage counters")
        print("  --print-env                 Show resolved config")
        print("  -c, --continue              Resume the most recent session")
        print("  --resume [sessionId]        Resume a session (interactive picker without an id)")
        print("  --sessions                  List saved sessions for this workspace")
        print("  --mode plan|code|full-access  Tool permissions; full-access lifts the turn limit")
        print("                              and runs the quality gate when the task ends")
        print("  --proxy-only                Run only the local proxy in foreground")
        print("  --help, -h                  Show this help")
        return

    load_config()
    port = config.PORT or find_free_port()

    if "--update-auggie" in launcher_args:
        sys.exit(update_auggie())

    if "--check" in launcher_args or "--doctor" in launcher_args:
        sys.exit(run_doctor(check_updates="--no-update-check" not in launcher_args))

    if "--mode" in launcher_args:
        mode = launcher_args[launcher_args.index("--mode") + 1]
        permissions = permissions_for_mode(mode)
        if permissions is None:
            print(f"error: unknown mode '{mode}' (expected: plan, code, full-access)", file=sys.stderr)
            sys.exit(2)
        # The flag is repeatable, one rule per occurrence -- passing five rules
        # after a single --permission makes the CLI see five arguments.
        for rule in permissions:
            pass_args.extend(["--permission", rule])
        # The proxy needs it too: a read-only turn must not remap a search onto
        # a mutating tool, which is what the CLI would then refuse.
        os.environ["AUGGIE_LAUNCH_MODE"] = mode.strip().lower()
        config.SESSION_MODE = mode.strip().lower()
        if mode_runs_until_done(mode):
            # Raise the ceiling the only way the CLI allows: its own flag. Passing
            # --max-turns cannot help, since it only accepts values below the
            # default and rejects 0 outright.
            config.AGENT_MAX_ITERATIONS = max(
                config.AGENT_MAX_ITERATIONS, config.FULL_ACCESS_MAX_ITERATIONS
            )
        log(f"mode {mode}: {len(permissions)} tool permission(s) applied")

    if "--stats" in launcher_args:
        from .stats import print_stats
        print_stats()
        return

    if "--reset-stats" in launcher_args:
        from .stats import reset
        reset()
        print("usage counters cleared")
        return

    if "--sessions" in launcher_args:
        print_sessions()
        return
    if "--models" in launcher_args:
        print_models()
        return

    if "--print-env" in launcher_args:
        print_env(port)
        return

    httpd = ThreadingHTTPServer(("127.0.0.1", port), AuggieProxy)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    proxy_url = f"http://127.0.0.1:{port}"

    if config.VERBOSE:
        log(f"proxy={proxy_url} upstream={upstream_url()} model={config.TARGET_MODEL} indexing={config.INDEXING_MODE}")
        if config._LOADED_ENV_FILES:
            log("env files: " + ", ".join(config._LOADED_ENV_FILES))

    if "--proxy-only" in launcher_args:
        try:
            print(f"Proxy ready at {proxy_url} (Forwarding to: {config.TARGET_BASE_URL})")
            thread.join()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                stop_server(httpd)
            except KeyboardInterrupt:
                pass
        return

    # Build full injections into Auggie CLI environment
    env = build_injected_environment(proxy_url)

    # Injected MCP config if enabled and not already provided
    locked, holder = acquire_session_lock()
    if not locked:
        # Advisory: the subscription is per-user, so a second session contends
        # for it. Warn rather than refuse, since a stale lock would otherwise
        # block a legitimate run.
        print(f"[auggie-launch] another session appears to be active ({holder}); "
              f"the subscription is per-user, so responses may contend",
              file=sys.stderr)

    if config.AUTO_INJECT_MCP and "--mcp-config" not in pass_args:
        mcp_path = generate_injected_mcp_config()
        if mcp_path and os.path.isfile(mcp_path):
            pass_args = ["--mcp-config", mcp_path, *pass_args]
            log(f"injected MCP config: {mcp_path}")

    exit_code = 0
    try:
        result = subprocess.run([config.AUGGIE_BIN, *pass_args], env=env)
        exit_code = result.returncode
    except FileNotFoundError:
        print(f"error: cannot find auggie binary ({config.AUGGIE_BIN})", file=sys.stderr)
        print("Tip: Ensure auggie is installed or run ./install.sh", file=sys.stderr)
        exit_code = 127
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        # A second Ctrl+C while shutting down must not abort the cleanup itself.
        try:
            stop_server(httpd)
        except KeyboardInterrupt:
            pass

    release_session_lock()

    if config.POST_RUN_CHECKS:
        # Runs after the proxy is down: the gate starts its own processes, and
        # leaving ours listening made the two collide.
        #
        # A Ctrl+C here means "I want out", not "check harder". The child gets
        # the interrupt too, so the run is abandoned rather than reported as a
        # failure -- an interrupted check says nothing about the code.
        from .server import post_run_checks
        try:
            gate_ok = report_post_run_checks(post_run_checks())
        except KeyboardInterrupt:
            print("\ninterrupted: quality gate skipped", file=sys.stderr)
            gate_ok = True
        if not gate_ok:
            exit_code = exit_code or 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
