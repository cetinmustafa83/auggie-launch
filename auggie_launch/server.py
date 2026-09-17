from __future__ import annotations

import socket
import threading
from http.server import ThreadingHTTPServer

from . import config
from .models import fetch_upstream_models, model_context_limit

# ============================================================================
# Local proxy server lifecycle & introspection
# ============================================================================


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
    from . import codegpt

    seen: set[str] = set()
    print("Configured model:")
    print(f"  - {config.TARGET_MODEL}")
    seen.add(config.TARGET_MODEL)

    if config.IS_CODEGPT:
        print("\nInclusive (economy) models:")
        for entry in codegpt.load_catalog_models():
            marker = " (active)" if entry["id"] == config.TARGET_MODEL else ""
            provider = entry.get("provider") or "?"
            print(f"  - {entry['id']}  [{provider}]{marker}")
            seen.add(entry["id"])
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
