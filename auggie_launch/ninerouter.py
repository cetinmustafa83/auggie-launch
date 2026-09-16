from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

from . import config
from .models import fetch_upstream_models, model_context_limit
from .proxy import extract_chat_text
from .upstream import (
    _ping_tunnel,
    get_active_key,
    json_bytes,
    log_router_response_headers,
    open_upstream_with_retries,
    upstream_headers,
)

# ============================================================================
# 9router Helper & CLI Diagnostics Commands
# ============================================================================

def print_9router_combos() -> None:
    """Lists all combos and their constituent models from 9router."""
    print("=" * 60)
    print("🔀 9ROUTER COMBOS & MODEL ALIASES")
    print("=" * 60)
    if config._LOCAL_9ROUTER.combos:
        print(f"Active Combos ({len(config._LOCAL_9ROUTER.combos)}):")
        for c in config._LOCAL_9ROUTER.combos:
            cname = c.get("name")
            models = c.get("models") or []
            print(f"  • {cname} ({len(models)} fallback models):")
            for m in models:
                print(f"      - {m}")
    else:
        print("  (No combos defined in ~/.9router/db.json)")

    if config._LOCAL_9ROUTER.model_aliases:
        print(f"\nModel Aliases ({len(config._LOCAL_9ROUTER.model_aliases)}):")
        for alias, real in config._LOCAL_9ROUTER.model_aliases.items():
            print(f"  • {alias:25} -> {real}")
    print("=" * 60)


def print_9router_stats() -> None:
    """Queries 9router's /api/usage and prints token savings and stats."""
    print("=" * 60)
    print("📊 9ROUTER USAGE & SAVINGS STATISTICS")
    print("=" * 60)

    # 1. Local state overview
    print(f"9router Installation : {'Found (~/.9router)' if config._LOCAL_9ROUTER.installed else 'Not detected'}")
    print(f"Caveman Mode         : {'Enabled' if config._LOCAL_9ROUTER.caveman_enabled else 'Disabled'} (Level: {config._LOCAL_9ROUTER.caveman_level})")
    print(f"Tunnel URL           : {config._LOCAL_9ROUTER.tunnel_url or '(none)'}")
    print(f"Configured Providers : {len(config._LOCAL_9ROUTER.provider_connections)} ({', '.join(config._LOCAL_9ROUTER.provider_api_keys.keys())})")

    # 2. Try querying live /api/usage
    parsed = urllib.parse.urlparse(config.TARGET_BASE_URL)
    base_host_url = f"{parsed.scheme}://{parsed.netloc}"
    usage_url = f"{base_host_url}/api/usage"

    try:
        req = urllib.request.Request(usage_url, headers=upstream_headers(get_active_key(), stream=False))
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("\nLive 9router Metrics:")
            print(f"  Raw Stats Response: {json.dumps(data, indent=2)}")
    except Exception as exc:
        print(f"\n(Live /api/usage not reachable: {exc})")

    print("=" * 60)


def start_9router_daemon() -> None:
    """Attempts to launch 9router if not running."""
    print("Starting 9router service...")
    binary = find_9router_binary()
    if not binary:
        print("9router binary not found; installing from npm...")
        if not install_9router():
            print("[FAIL] Could not install 9router. Install manually: npm i -g 9router@latest")
            return
        binary = find_9router_binary()

    restore_9router_db()

    if not binary:
        print("[FAIL] Could not launch 9router binary. Start it manually via: 9router")
        return

    try:
        subprocess.Popen([binary, "--skip-update"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[FAIL] Could not launch 9router ({binary}): {exc}")
        return

    print(f"[OK] Launched 9router process via: {binary}")
    print("Waiting 2 seconds for socket...")
    time.sleep(2.0)


# ============================================================================
# 9router Installation, Update, and Database Restore
# ============================================================================

_9ROUTER_NPM_PACKAGE = "9router@latest"
_BUNDLED_DB_BACKUP_DIR = os.path.join(config._HERE, "9router", "db")


def find_9router_binary() -> str | None:
    """Locates the installed 9router CLI, or None if it is not installed."""
    found = shutil.which("9router")
    if found:
        return found
    home = os.path.expanduser("~")
    for cand in (
        os.path.join(home, ".npm-global", "bin", "9router"),
        os.path.join(home, ".bun", "bin", "9router"),
        "/usr/local/bin/9router",
        "/opt/homebrew/bin/9router",
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def install_9router(*, update: bool = False) -> bool:
    """Installs or updates 9router globally via npm. Returns True on success."""
    npm = shutil.which("npm")
    if not npm:
        print("[FAIL] npm not found in PATH; install Node.js first (https://nodejs.org)")
        return False

    action = "Updating" if update else "Installing"
    print(f"{action} {_9ROUTER_NPM_PACKAGE} (npm i -g {_9ROUTER_NPM_PACKAGE} --prefer-online)...")
    try:
        result = subprocess.run(
            [npm, "i", "-g", _9ROUTER_NPM_PACKAGE, "--prefer-online"],
            check=False,
        )
    except Exception as exc:
        print(f"[FAIL] npm install failed: {exc}")
        return False

    if result.returncode != 0:
        print(f"[FAIL] npm exited with code {result.returncode}")
        return False

    binary = find_9router_binary()
    print(f"[OK] 9router {'updated' if update else 'installed'}: {binary or '(binary not on PATH yet)'}")
    return True


def latest_bundled_db_backup() -> str | None:
    """Newest 9router DB backup shipped in the repo's 9router/db directory."""
    try:
        backups = [
            os.path.join(_BUNDLED_DB_BACKUP_DIR, name)
            for name in os.listdir(_BUNDLED_DB_BACKUP_DIR)
            if name.endswith(".json")
        ]
    except OSError:
        return None
    if not backups:
        return None
    return max(backups, key=os.path.getmtime)


def restore_9router_db(*, force: bool = False) -> bool:
    """Restores ~/.9router/db.json from the bundled backup.

    Only restores when the live DB is missing, unless force=True (then the
    existing DB is copied aside to db.json.bak first). Returns True if restored.
    """
    backup = latest_bundled_db_backup()
    if not backup:
        if force:
            print(f"[FAIL] No bundled DB backup found in {_BUNDLED_DB_BACKUP_DIR}")
        return False

    nine_dir = os.path.join(os.path.expanduser("~"), ".9router")
    db_file = os.path.join(nine_dir, "db.json")

    if os.path.isfile(db_file) and not force:
        return False

    try:
        with open(backup, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or "settings" not in payload:
            print(f"[FAIL] Backup does not look like a 9router DB: {backup}")
            return False

        os.makedirs(nine_dir, exist_ok=True)
        if os.path.isfile(db_file):
            shutil.copy2(db_file, db_file + ".bak")
            print(f"[OK] Existing DB saved to {db_file}.bak")
        shutil.copy2(backup, db_file)
    except Exception as exc:
        print(f"[FAIL] Could not restore 9router DB: {exc}")
        return False

    print(f"[OK] Restored 9router DB from {os.path.basename(backup)} -> {db_file}")
    return True


def ensure_9router_installed() -> bool:
    """Installs 9router when the system cannot detect it, and seeds its DB."""
    binary = find_9router_binary()
    if binary and config._LOCAL_9ROUTER.installed:
        return True

    if not binary:
        print("9router not detected on this system.")
        if not install_9router():
            return False

    restore_9router_db()
    return find_9router_binary() is not None


def run_doctor_check() -> int:
    """Performs deep health and connectivity verification for 9router and Auggie."""
    print("=" * 60)
    print("🔍 AUGGIE-LAUNCH HEALTH & 9ROUTER DIAGNOSTICS")
    print("=" * 60)

    # 1. Config summary
    print(f"Target Base URL : {config.TARGET_BASE_URL}")
    print(f"Target Model    : {config.TARGET_MODEL}")
    print(f"API Key         : {config.API_KEYS[0][:10]}... ({len(config.API_KEYS)} key(s) loaded)")
    print(f"9router Detected: {'YES (native local + host match)' if config.IS_9ROUTER else 'NO'}")
    print(f"Local 9router DB: {'Found (~/.9router/db.json)' if config._LOCAL_9ROUTER.installed else 'Not found'}")
    if config._LOCAL_9ROUTER.tunnel_url:
        print(f"Cloudflare Tunl : {config._LOCAL_9ROUTER.tunnel_url}")
    print(f"Caveman Mode    : {'Enabled (' + config.ROUTER_CAVEMAN_LEVEL + ')' if config.ROUTER_CAVEMAN_MODE else 'Disabled'}")
    print(f"Reasoning Stream: {config.STREAM_THINKING}")
    print("-" * 60)

    # 2. Upstream Network & Ping Test
    print("1. Testing connection to Upstream / 9router...")
    parsed = urllib.parse.urlparse(config.TARGET_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    connected = False
    try:
        sock.connect((host, port))
        sock.close()
        connected = True
        print(f"   [OK] Socket connection succeeded to {host}:{port}")
    except Exception as exc:
        print(f"   [WARN] Cannot connect to local {host}:{port}: {exc}")
        # Tunnel fallback ping
        if config._LOCAL_9ROUTER.tunnel_url:
            if _ping_tunnel(config._LOCAL_9ROUTER.tunnel_url):
                t_parsed = urllib.parse.urlparse(config._LOCAL_9ROUTER.tunnel_url)
                t_host = t_parsed.hostname or "localhost"
                t_port = t_parsed.port or (443 if t_parsed.scheme == "https" else 80)
                print(f"   [OK] Tunnel reachable: {t_host}:{t_port}")
            else:
                print("   [WARN] Tunnel also unreachable")

    # 3. Model Registry Discovery
    print("2. Querying Model Registry (/models)...")
    try:
        models = fetch_upstream_models()
        if models:
            print(f"   [OK] Discovered {len(models)} models available from 9router:")
            for m in models[:6]:
                mid = m.get("id") or m.get("name")
                print(f"        - {mid}")
            if len(models) > 6:
                print(f"        ... and {len(models) - 6} more.")
        else:
            print(f"   [INFO] Discovered {len(config._LOCAL_9ROUTER.combos)} combos & {len(config._LOCAL_9ROUTER.model_aliases)} aliases in local 9router DB.")
    except Exception as exc:
        print(f"   [WARN] Could not query /models: {exc}")

    # 4. Minimal Completion Test
    if connected:
        print(f"3. Testing completion ping with model '{config.TARGET_MODEL}'...")
        req_body = {
            "model": config.TARGET_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        t0 = time.time()
        try:
            with open_upstream_with_retries(json_bytes(req_body), stream=False, timeout=15, label="doctor") as resp:
                raw = resp.read()
                latency = (time.time() - t0) * 1000
                data = json.loads(raw.decode("utf-8"))
                text = extract_chat_text(data)
                print(f"   [OK] Upstream completion response in {latency:.1f}ms: {text!r}")
                log_router_response_headers(resp.headers)
        except Exception as exc:
            print(f"   [WARN] Completion request warning: {exc}")
    else:
        print("3. Skipping completion ping (socket offline)")

    # 5. Auggie CLI binary check
    print("4. Checking Auggie CLI binary...")
    try:
        res = subprocess.run([config.AUGGIE_BIN, "--version"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            print(f"   [OK] Auggie binary found: {config.AUGGIE_BIN} -> {res.stdout.strip()}")
        else:
            print(f"   [WARN] Auggie returned exit code {res.returncode}: {res.stderr.strip()}")
    except FileNotFoundError:
        print(f"   [FAIL] Auggie CLI '{config.AUGGIE_BIN}' not found on PATH.")
        print("   💡 Install with: npm install -g @augmentcode/auggie or run ./install.sh")
    except Exception as exc:
        print(f"   [WARN] Error checking auggie: {exc}")

    print("=" * 60)
    print("✅ System diagnostics completed!")
    return 0


def print_models() -> None:
    models = fetch_upstream_models()
    seen = set()
    print("Available models in 9router & Upstream:")
    if config._LOCAL_9ROUTER.combos:
        print("  [9router Combos]:")
        for c in config._LOCAL_9ROUTER.combos:
            cname = c.get("name")
            seen.add(cname)
            print(f"    - {cname}")
    if config._LOCAL_9ROUTER.model_aliases:
        print("  [9router Aliases]:")
        for a, r in config._LOCAL_9ROUTER.model_aliases.items():
            seen.add(a)
            print(f"    - {a} -> {r}")
    if models:
        print("  [Live Registry Models]:")
        for m in models:
            mid = m.get("id") or m.get("name")
            if mid not in seen:
                print(f"    - {mid}")


def print_env(port: int) -> None:
    print(f"AUGGIE_BIN={config.AUGGIE_BIN}")
    print(f"AUGGIE_LAUNCH_PROXY_URL=http://127.0.0.1:{port}")
    print(f"AUGGIE_LAUNCH_BASE_URL={config.TARGET_BASE_URL}")
    print(f"AUGGIE_LAUNCH_MODEL={config.TARGET_MODEL}")
    print(f"AUGGIE_LAUNCH_API_KEY={'<set>' if config.API_KEYS else ''}")
    print(f"AUGGIE_LAUNCH_IS_9ROUTER={'true' if config.IS_9ROUTER else 'false'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_LOCAL_DB={'true' if config._LOCAL_9ROUTER.installed else 'false'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_CAVEMAN={'true (' + config.ROUTER_CAVEMAN_LEVEL + ')' if config.ROUTER_CAVEMAN_MODE else 'false'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_TUNNEL={config._LOCAL_9ROUTER.tunnel_url or '(none)'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_COMBOS={','.join(str(c.get('name')) for c in config._LOCAL_9ROUTER.combos if c.get('name')) or '(none)'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_ALIASES={','.join(config._LOCAL_9ROUTER.model_aliases.keys()) or '(none)'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_PROVIDERS={','.join(config._LOCAL_9ROUTER.provider_api_keys.keys()) or '(none)'}")
    print(f"AUGGIE_LAUNCH_INDEXING_MODE={config.INDEXING_MODE}")
    print(f"AUGGIE_LAUNCH_STREAM_THINKING={'true' if config.STREAM_THINKING else 'false'}")
    print(f"AUGGIE_LAUNCH_USER_AGENT={config.UPSTREAM_USER_AGENT}")
    print(f"AUGGIE_LAUNCH_UPSTREAM_APP_NAME={config.UPSTREAM_APP_NAME}")
    print(f"AUGGIE_LAUNCH_MODEL_CONTEXT={model_context_limit(config.TARGET_MODEL)}")
    print("loaded_env_files=" + (", ".join(config._LOADED_ENV_FILES) if config._LOADED_ENV_FILES else "(none)"))


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


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


