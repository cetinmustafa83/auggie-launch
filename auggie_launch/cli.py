from __future__ import annotations

import os
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer

from . import config
from .config import load_config, log
from .injections import build_injected_environment, generate_injected_mcp_config
from .ninerouter import (
    find_free_port,
    install_9router,
    print_9router_combos,
    print_9router_stats,
    print_env,
    print_models,
    restore_9router_db,
    run_doctor_check,
    start_9router_daemon,
    stop_server,
)
from .proxy import AuggieProxy
from .upstream import upstream_url

# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    launcher_args = []
    pass_args = []
    args = sys.argv[1:]
    while args:
        arg = args.pop(0)
        if arg == "--":
            pass_args.extend(args)
            break
        if arg in {
            "--print-env",
            "--proxy-only",
            "--check",
            "--9router-doctor",
            "--doctor",
            "--models",
            "--combos",
            "--stats",
            "--usage",
            "--start-9router",
            "--install-9router",
            "--update-9router",
            "--restore-9router-db",
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
        print("  --check, --9router-doctor   Test connectivity, 9router health, and models")
        print("  --models                    List all models discovered from 9router")
        print("  --combos                    List 9router combos and fallback groups")
        print("  --stats, --usage            Show 9router token savings and provider status")
        print("  --start-9router             Start 9router (installs it via npm if missing)")
        print("  --install-9router           Install 9router globally (npm i -g 9router@latest)")
        print("  --update-9router            Update 9router to the latest npm release")
        print("  --restore-9router-db        Restore ~/.9router/db.json from the bundled backup")
        print("  --print-env                 Show resolved config")
        print("  --proxy-only                Run only the local proxy in foreground")
        print("  --help, -h                  Show this help")
        return

    if "--install-9router" in launcher_args:
        ok = install_9router()
        restore_9router_db()
        sys.exit(0 if ok else 1)

    if "--update-9router" in launcher_args:
        sys.exit(0 if install_9router(update=True) else 1)

    if "--restore-9router-db" in launcher_args:
        sys.exit(0 if restore_9router_db(force=True) else 1)

    load_config()
    port = config.PORT or find_free_port()

    if "--combos" in launcher_args:
        print_9router_combos()
        return

    if "--stats" in launcher_args or "--usage" in launcher_args:
        print_9router_stats()
        return

    if "--start-9router" in launcher_args:
        start_9router_daemon()
        return

    if "--check" in launcher_args or "--9router-doctor" in launcher_args or "--doctor" in launcher_args:
        sys.exit(run_doctor_check())

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
        log(f"proxy={proxy_url} upstream={upstream_url()} model={config.TARGET_MODEL} 9router={config.IS_9ROUTER} indexing={config.INDEXING_MODE}")
        if config._LOADED_ENV_FILES:
            log("env files: " + ", ".join(config._LOADED_ENV_FILES))

    if "--proxy-only" in launcher_args:
        try:
            print(f"Proxy ready at {proxy_url} (Forwarding to: {config.TARGET_BASE_URL})")
            thread.join()
        except KeyboardInterrupt:
            pass
        finally:
            stop_server(httpd)
        return

    # Build full injections into Auggie CLI environment
    env = build_injected_environment(proxy_url)

    # Injected MCP config if enabled and not already provided
    if config.AUTO_INJECT_MCP and "--mcp-config" not in pass_args:
        mcp_path = generate_injected_mcp_config()
        if mcp_path and os.path.isfile(mcp_path):
            pass_args = ["--mcp-config", mcp_path, *pass_args]
            log(f"injected MCP tools config from 9router: {mcp_path}")

    try:
        result = subprocess.run([config.AUGGIE_BIN, *pass_args], env=env)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"error: cannot find auggie binary ({config.AUGGIE_BIN})", file=sys.stderr)
        print("Tip: Ensure auggie is installed or run ./install.sh", file=sys.stderr)
        sys.exit(127)
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        stop_server(httpd)


if __name__ == "__main__":
    main()
