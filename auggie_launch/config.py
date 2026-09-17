#!/usr/bin/env python3
# Copyright (c) 2026 auggie-launch contributors
# For learning and research only; any other use is at your own risk.
"""auggie-launch: local Python proxy and launcher for Auggie/Augment Code CLI.

catalog, tunnel, and provider keys), provides full injections into Auggie CLI
(session auth, environment, MCP tools, Caveman ultra instructions, feature flags),
and forwards chat/completion requests to OpenAI/Claude-compatible upstream endpoints.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import threading
import uuid
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))

_REQUIRED_KEYS = (
    "AUGGIE_LAUNCH_BASE_URL",
    "AUGGIE_LAUNCH_MODEL",
)
_MANAGED_ENV_PREFIXES = ("AUGGIE_LAUNCH_",)
_MANAGED_ENV_KEYS = {"AUGGIE_BIN"}

__version__ = "0.4.0"

# --- Global Configurations ---
TARGET_BASE_URL = ""
# Runtime upstream override (single upstream today)
# Cloudflare tunnel takes over. Sticky for the process lifetime.
# lc-debt: no automatic switch back to local once the tunnel takes over; restart to reset.
ACTIVE_BASE_URL = ""
BASE_URL_LOCK = threading.Lock()
TARGET_MODEL = ""
TARGET_API_KEY = ""
API_KEYS: list[str] = []
FROZEN_KEYS: dict[str, float] = {}
FROZEN_LOCK = threading.Lock()
AUGGIE_BIN = "auggie"
AUGGIE_PACKAGE = "@augmentcode/auggie"
LOCAL_TOKEN = "fake-augment-access-token"
_LOADED_ENV_FILES: list[str] = []
# Stable per-process identifiers for the Codex-style upstream headers.
_CODEX_SESSION_ID = f"session-{uuid.uuid4()}"
_CODEX_THREAD_ID = str(uuid.uuid4())
_CODEX_INSTALLATION_ID = str(uuid.uuid4())
VERBOSE = False
DEBUG_DIR = ""
# Debug dumps older than this are pruned on the next write.
DEBUG_RETENTION_SECONDS = 3600.0
PORT = 0

UPSTREAM_USER_AGENT = "codex-cli"
UPSTREAM_APP_NAME = "Codex"
UPSTREAM_MIN_INTERVAL_SECONDS = 0.25
UPSTREAM_RETRIES = 2
UPSTREAM_429_FREEZE_SECONDS = 60.0
UPSTREAM_5XX_FREEZE_SECONDS = 30.0
UPSTREAM_MAX_RETRY_AFTER_SECONDS = 300.0
UPSTREAM_BACKOFF_INITIAL_SECONDS = 1.0
UPSTREAM_BACKOFF_MAX_SECONDS = 30.0
UPSTREAM_TIMEOUT_SECONDS = 300.0
UPSTREAM_NEXT_REQUEST_AT = 0.0
UPSTREAM_COOLDOWN_UNTIL = 0.0
UPSTREAM_THROTTLE_LOCK = threading.Lock()

SANITIZE_UPSTREAM_PROMPTS = False
# Force replies into one language ("English"), or "" to leave it to the model.
REPLY_LANGUAGE = ""
INDEXING_MODE = "complete"
MODEL_CONTEXT_TOKENS = 200000
MODEL_CONTEXT_TOKENS_EXPLICIT = False
MODEL_MAX_OUTPUT_TOKENS = 16000
REQUIRE_LOCAL_TOKEN = True
REASONING_EFFORT = ""
# Raw  tags are rendered verbatim by the CLI and end up in the transcript,
# which also confuses the model on later turns. The CLI renders reasoning by
# itself, so this stays off unless explicitly enabled.
STREAM_THINKING = False

# --- Auggie history summarization (session compaction) ---
HISTORY_SUMMARY_ENABLED = True
HISTORY_SUMMARY_MIN_VERSION = "0.0.1"
# Trigger point for history summarization. When the env var is unset the value
# is derived from the model's own window via HISTORY_SUMMARY_TRIGGER_RATIO.
HISTORY_SUMMARY_TRIGGER_TOKENS = 120000
HISTORY_SUMMARY_TRIGGER_EXPLICIT = False
HISTORY_SUMMARY_TRIGGER_RATIO = 0.6
HISTORY_SUMMARY_MAX_HISTORY_CHARS = 100000
HISTORY_SUMMARY_MAX_HISTORY_EXPLICIT = False
HISTORY_SUMMARY_INPUT_BUDGET_RATIO = 0.6

# --- Upstream model discovery ---
DYNAMIC_MODELS = True
USE_COMPLETION_TOKENS = "auto"  # auto | true | false
ENABLE_CONNECTION_POOL = True
AUTO_INJECT_MCP = True
ENABLE_PLAN_MODE = True
# Session mode chosen with --mode; read-only when it is "plan".
SESSION_MODE = ""
# Run the project quality gate after the CLI exits.
POST_RUN_CHECKS = True
# Turn ceiling handed to the CLI. Its default is 200; full-access raises it so a
# long task is not cut off mid-way.
AGENT_MAX_ITERATIONS = 200
FULL_ACCESS_MAX_ITERATIONS = 10000
ENABLE_PERSONA = True
# Cached catalog of upstream model metadata (context windows, capabilities)
CACHED_CATALOG: dict[str, Any] = {}
_CACHED_MODELS: list[dict[str, Any]] = []
_CACHED_MODELS_TIME = 0.0
_MODELS_CACHE_TTL = 60.0
_MODELS_LOCK = threading.Lock()
# --- CodeGPT Plus (agent-backed cloud) specific features ---
IS_CODEGPT = False
CODEGPT_SESSION_URL = ""
CODEGPT_TOKEN = ""
CODEGPT_ORG_ID = ""
CODEGPT_DISTINCT_ID = ""
CODEGPT_SIGNED_DISTINCT_ID = ""
CODEGPT_VERSION = "3.24.70"
# Inclusive-model bridge: the endpoint is /chat/tools/<harness> and the model's
# upstream must be named in an X-Provider header. No agent is involved.
CODEGPT_HARNESS = "codegpt"
CODEGPT_PROVIDER = ""
CODEGPT_SESSION_ID = ""



# ============================================================================
# Environment & Configuration Loading
# ============================================================================

def _parse_dotenv(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                out[key] = value
    except OSError:
        pass
    return out


def _candidate_env_paths() -> list[str]:
    paths: list[str] = []
    explicit = os.environ.get("AUGGIE_LAUNCH_ENV")
    if explicit:
        paths.append(os.path.expanduser(explicit))

    cwd = os.getcwd()
    paths.append(os.path.join(cwd, ".env"))
    paths.append(os.path.join(cwd, ".auggie-launch.env"))

    parent = os.path.dirname(cwd)
    for _ in range(6):
        if not parent or parent == os.path.dirname(parent):
            break
        paths.append(os.path.join(parent, ".env"))
        paths.append(os.path.join(parent, ".auggie-launch.env"))
        parent = os.path.dirname(parent)

    paths.append(os.path.join(_HERE, ".env"))
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    paths.append(os.path.join(xdg, "auggie-launch", ".env"))
    paths.append(os.path.expanduser("~/.auggie-launch.env"))

    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        ap = os.path.abspath(path)
        if ap not in seen:
            seen.add(ap)
            unique.append(ap)
    return unique


def _is_managed_env_key(key: str) -> bool:
    return key in _MANAGED_ENV_KEYS or key.startswith(_MANAGED_ENV_PREFIXES)


def load_dotenv_files() -> list[str]:
    loaded: list[str] = []
    claimed = {key for key in os.environ.keys() if not _is_managed_env_key(key)}
    for path in _candidate_env_paths():
        if not os.path.isfile(path):
            continue
        data = _parse_dotenv(path)
        if not data:
            continue
        for key, value in data.items():
            if key in claimed:
                continue
            os.environ[key] = value
            claimed.add(key)
        loaded.append(path)
    return loaded


def env_truthy(name: str, fallback: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


def bounded_float(value: str | None, default: float, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value) if value is not None and value.strip() else default
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def bounded_int(value: str | None, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value) if value is not None and value.strip() else default
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def env_int(name: str, fallback: int) -> int:
    try:
        value = int((os.environ.get(name) or "").strip())
        return value if value > 0 else fallback
    except ValueError:
        return fallback


def codex_version() -> str:
    configured = (os.environ.get("CODEX_VERSION") or "").strip()
    if configured:
        return configured
    codex_bin = (os.environ.get("CODEX_BIN") or "codex").strip() or "codex"
    try:
        result = subprocess.run(
            [codex_bin, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return "0.144.6"
    for token in result.stdout.replace("v", " v").split():
        candidate = token.lstrip("v")
        parts = candidate.split(".")
        if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
            return candidate
    return "0.144.6"


def codex_user_agent() -> str:
    originator = (os.environ.get("CODEX_ORIGINATOR") or "codex_cli_rs").strip() or "codex_cli_rs"
    version = codex_version()
    system = platform.system() or "Unknown"
    release = platform.release() or "unknown"
    arch = platform.machine() or "unknown"
    return f"{originator}/{version} ({system} {release}; {arch})"


def default_upstream_user_agent(fallback: str) -> str:
    if env_truthy("CODEX_HEAD"):
        return (os.environ.get("CODEX_USER_AGENT") or codex_user_agent()).strip()
    return fallback


def with_codex_headers(headers: dict[str, str]) -> dict[str, str]:
    if not env_truthy("CODEX_HEAD"):
        return headers

    out = dict(headers)
    out["User-Agent"] = out.get("User-Agent") or codex_user_agent()
    out.setdefault("Originator", (os.environ.get("CODEX_ORIGINATOR") or "codex_cli_rs").strip() or "codex_cli_rs")
    out.setdefault("session-id", (os.environ.get("CODEX_SESSION_ID") or _CODEX_SESSION_ID).strip())
    out.setdefault("thread-id", (os.environ.get("CODEX_THREAD_ID") or _CODEX_THREAD_ID).strip())
    out.setdefault("x-codex-installation-id", (os.environ.get("CODEX_INSTALLATION_ID") or _CODEX_INSTALLATION_ID).strip())
    beta = (os.environ.get("CODEX_OPENAI_BETA") or "").strip()
    if beta:
        out.setdefault("OpenAI-Beta", beta)
    return out


def load_config() -> None:
    global TARGET_BASE_URL, TARGET_MODEL, TARGET_API_KEY, API_KEYS, AUGGIE_BIN, AUGGIE_PACKAGE
    global LOCAL_TOKEN, VERBOSE, DEBUG_DIR, PORT, _LOADED_ENV_FILES, DEBUG_RETENTION_SECONDS
    global UPSTREAM_USER_AGENT, UPSTREAM_APP_NAME, SANITIZE_UPSTREAM_PROMPTS
    global REPLY_LANGUAGE, STREAM_THINKING, DYNAMIC_MODELS, USE_COMPLETION_TOKENS
    global ENABLE_CONNECTION_POOL, AUTO_INJECT_MCP, CACHED_CATALOG
    global ENABLE_PLAN_MODE, ENABLE_PERSONA, SESSION_MODE, POST_RUN_CHECKS
    global AGENT_MAX_ITERATIONS, FULL_ACCESS_MAX_ITERATIONS
    global UPSTREAM_MIN_INTERVAL_SECONDS, UPSTREAM_RETRIES
    global UPSTREAM_429_FREEZE_SECONDS, UPSTREAM_5XX_FREEZE_SECONDS
    global UPSTREAM_MAX_RETRY_AFTER_SECONDS, UPSTREAM_BACKOFF_INITIAL_SECONDS
    global UPSTREAM_BACKOFF_MAX_SECONDS
    global INDEXING_MODE, MODEL_CONTEXT_TOKENS, MODEL_MAX_OUTPUT_TOKENS, REASONING_EFFORT
    global MODEL_CONTEXT_TOKENS_EXPLICIT, REQUIRE_LOCAL_TOKEN
    global UPSTREAM_TIMEOUT_SECONDS
    global DYNAMIC_MODELS, USE_COMPLETION_TOKENS, ENABLE_CONNECTION_POOL
    global CACHED_CATALOG, AUTO_INJECT_MCP

    _LOADED_ENV_FILES = load_dotenv_files()

    # Read model catalog for capabilities (vision, audio, pdf, context window)
    base_url_env = os.environ.get("AUGGIE_LAUNCH_BASE_URL", "").strip()
    if not base_url_env:
        base_url_env = "http://localhost:20128/v1"
        os.environ["AUGGIE_LAUNCH_BASE_URL"] = base_url_env

    model_env = os.environ.get("AUGGIE_LAUNCH_MODEL", "").strip()
    if not model_env:
        model_env = "free"
        os.environ["AUGGIE_LAUNCH_MODEL"] = model_env

    key_text = os.environ.get("AUGGIE_LAUNCH_API_KEYS") or os.environ.get("AUGGIE_LAUNCH_API_KEY") or ""

    API_KEYS = [key.strip() for key in key_text.split(",") if key.strip()]

    missing = [key for key in _REQUIRED_KEYS if not (os.environ.get(key) or "").strip()]
    if not API_KEYS:
        missing.append("AUGGIE_LAUNCH_API_KEY")
    if missing:
        xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        print("error: missing required configuration: " + ", ".join(missing), file=sys.stderr)
        print(f"Set them in ./.env or {os.path.join(xdg, 'auggie-launch', '.env')}", file=sys.stderr)
        print(f"template: {os.path.join(_HERE, '.env.example')}", file=sys.stderr)
        sys.exit(2)

    TARGET_BASE_URL = os.environ["AUGGIE_LAUNCH_BASE_URL"].strip().rstrip("/")
    TARGET_MODEL = os.environ["AUGGIE_LAUNCH_MODEL"].strip()
    TARGET_API_KEY = API_KEYS[0]
    AUGGIE_BIN = (os.environ.get("AUGGIE_BIN") or "auggie").strip()
    AUGGIE_PACKAGE = (os.environ.get("AUGGIE_LAUNCH_AUGGIE_PACKAGE") or "@augmentcode/auggie").strip()
    # Per-session random token unless pinned: the proxy is loopback-only but still
    # refuses requests that do not carry the token it injected into Auggie.
    LOCAL_TOKEN = (os.environ.get("AUGGIE_LAUNCH_LOCAL_TOKEN") or f"al-{uuid.uuid4().hex}").strip()
    VERBOSE = env_truthy("AUGGIE_LAUNCH_VERBOSE")
    DEBUG_DIR = (os.environ.get("AUGGIE_LAUNCH_DEBUG_DIR") or os.path.join(tempfile.gettempdir(), "auggie-launch")).strip()
    DEBUG_RETENTION_SECONDS = bounded_float(os.environ.get("AUGGIE_LAUNCH_DEBUG_RETENTION_SECONDS"), 3600.0)
    PORT = env_int("AUGGIE_LAUNCH_PORT", 0)

    UPSTREAM_USER_AGENT = (os.environ.get("AUGGIE_LAUNCH_USER_AGENT") or default_upstream_user_agent("codex-cli")).strip()
    UPSTREAM_APP_NAME = (os.environ.get("AUGGIE_LAUNCH_UPSTREAM_APP_NAME") or "Codex").strip()
    UPSTREAM_MIN_INTERVAL_SECONDS = bounded_float(os.environ.get("AUGGIE_LAUNCH_UPSTREAM_MIN_INTERVAL_SECONDS"), 0.25)
    UPSTREAM_RETRIES = bounded_int(os.environ.get("AUGGIE_LAUNCH_UPSTREAM_RETRIES"), 2)
    UPSTREAM_429_FREEZE_SECONDS = bounded_float(os.environ.get("AUGGIE_LAUNCH_429_FREEZE_SECONDS"), 60.0)
    UPSTREAM_5XX_FREEZE_SECONDS = bounded_float(os.environ.get("AUGGIE_LAUNCH_5XX_FREEZE_SECONDS"), 30.0)
    UPSTREAM_MAX_RETRY_AFTER_SECONDS = bounded_float(os.environ.get("AUGGIE_LAUNCH_MAX_RETRY_AFTER_SECONDS"), 300.0)
    UPSTREAM_BACKOFF_INITIAL_SECONDS = bounded_float(os.environ.get("AUGGIE_LAUNCH_BACKOFF_INITIAL_SECONDS"), 1.0)
    UPSTREAM_BACKOFF_MAX_SECONDS = bounded_float(os.environ.get("AUGGIE_LAUNCH_BACKOFF_MAX_SECONDS"), 30.0)
    UPSTREAM_TIMEOUT_SECONDS = bounded_float(os.environ.get("AUGGIE_LAUNCH_UPSTREAM_TIMEOUT"), 300.0)
    SANITIZE_UPSTREAM_PROMPTS = env_truthy("AUGGIE_LAUNCH_SANITIZE_UPSTREAM_PROMPTS", False)
    REPLY_LANGUAGE = (os.environ.get("AUGGIE_LAUNCH_REPLY_LANGUAGE") or "").strip()
    INDEXING_MODE = (os.environ.get("AUGGIE_LAUNCH_INDEXING_MODE") or "complete").strip().lower()
    MODEL_CONTEXT_TOKENS = env_int("AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS", 200000)
    MODEL_CONTEXT_TOKENS_EXPLICIT = bool((os.environ.get("AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS") or "").strip())
    REQUIRE_LOCAL_TOKEN = env_truthy("AUGGIE_LAUNCH_REQUIRE_LOCAL_TOKEN", True)
    MODEL_MAX_OUTPUT_TOKENS = env_int("AUGGIE_LAUNCH_MODEL_MAX_OUTPUT_TOKENS", 16000)
    REASONING_EFFORT = (os.environ.get("AUGGIE_LAUNCH_REASONING_EFFORT") or "").strip().lower()
    if REASONING_EFFORT and REASONING_EFFORT not in {"low", "medium", "high"}:
        print("error: AUGGIE_LAUNCH_REASONING_EFFORT must be low, medium, or high", file=sys.stderr)
        sys.exit(2)

    STREAM_THINKING = env_truthy("AUGGIE_LAUNCH_STREAM_THINKING", False)
    # Runtime upstream override (single upstream today; kept for failover)
    global ACTIVE_BASE_URL
    ACTIVE_BASE_URL = ""

    # CodeGPT Plus: agent-backed cloud that speaks an OpenAI-shaped SSE stream.
    global IS_CODEGPT, CODEGPT_SESSION_URL, CODEGPT_TOKEN, CODEGPT_ORG_ID
    global CODEGPT_DISTINCT_ID, CODEGPT_SIGNED_DISTINCT_ID, CODEGPT_VERSION
    global CODEGPT_HARNESS, CODEGPT_PROVIDER, CODEGPT_SESSION_ID
    global HISTORY_SUMMARY_ENABLED, HISTORY_SUMMARY_MIN_VERSION, HISTORY_SUMMARY_TRIGGER_TOKENS
    global HISTORY_SUMMARY_TRIGGER_EXPLICIT, HISTORY_SUMMARY_TRIGGER_RATIO
    global HISTORY_SUMMARY_MAX_HISTORY_CHARS, HISTORY_SUMMARY_MAX_HISTORY_EXPLICIT
    global HISTORY_SUMMARY_INPUT_BUDGET_RATIO
    # Unset -> default sidecar URL. Set-but-empty -> disable sidecar probing (pinned token).
    _session_env = os.environ.get("AUGGIE_LAUNCH_CODEGPT_SESSION_URL")
    CODEGPT_SESSION_URL = "http://localhost:54112/api/session" if _session_env is None else _session_env.strip()

    HISTORY_SUMMARY_ENABLED = env_truthy("AUGGIE_LAUNCH_HISTORY_SUMMARY", True)
    HISTORY_SUMMARY_MIN_VERSION = (os.environ.get("AUGGIE_LAUNCH_HISTORY_SUMMARY_MIN_VERSION") or "0.0.1").strip()
    HISTORY_SUMMARY_TRIGGER_TOKENS = env_int("AUGGIE_LAUNCH_HISTORY_SUMMARY_TRIGGER_TOKENS", 120000)
    HISTORY_SUMMARY_TRIGGER_EXPLICIT = bool((os.environ.get("AUGGIE_LAUNCH_HISTORY_SUMMARY_TRIGGER_TOKENS") or "").strip())
    HISTORY_SUMMARY_TRIGGER_RATIO = bounded_float(os.environ.get("AUGGIE_LAUNCH_HISTORY_SUMMARY_TRIGGER_RATIO"), 0.6)
    HISTORY_SUMMARY_MAX_HISTORY_CHARS = env_int("AUGGIE_LAUNCH_HISTORY_SUMMARY_MAX_HISTORY_CHARS", 100000)
    HISTORY_SUMMARY_MAX_HISTORY_EXPLICIT = bool((os.environ.get("AUGGIE_LAUNCH_HISTORY_SUMMARY_MAX_HISTORY_CHARS") or "").strip())
    HISTORY_SUMMARY_INPUT_BUDGET_RATIO = bounded_float(os.environ.get("AUGGIE_LAUNCH_HISTORY_SUMMARY_INPUT_BUDGET_RATIO"), 0.6)
    CODEGPT_TOKEN = (os.environ.get("AUGGIE_LAUNCH_CODEGPT_TOKEN") or "").strip()
    CODEGPT_ORG_ID = (os.environ.get("AUGGIE_LAUNCH_CODEGPT_ORG_ID") or "").strip()
    CODEGPT_DISTINCT_ID = (os.environ.get("AUGGIE_LAUNCH_CODEGPT_DISTINCT_ID") or "").strip()
    CODEGPT_SIGNED_DISTINCT_ID = (os.environ.get("AUGGIE_LAUNCH_CODEGPT_SIGNED_DISTINCT_ID") or "").strip()
    CODEGPT_VERSION = (os.environ.get("AUGGIE_LAUNCH_CODEGPT_VERSION") or "3.24.70").strip()
    CODEGPT_HARNESS = (os.environ.get("AUGGIE_LAUNCH_CODEGPT_HARNESS") or "codegpt").strip()
    CODEGPT_PROVIDER = (os.environ.get("AUGGIE_LAUNCH_CODEGPT_PROVIDER") or "").strip()
    CODEGPT_SESSION_ID = (os.environ.get("AUGGIE_LAUNCH_CODEGPT_SESSION_ID") or "").strip()
    IS_CODEGPT = detect_codegpt(TARGET_BASE_URL)
    DYNAMIC_MODELS = env_truthy("AUGGIE_LAUNCH_DYNAMIC_MODELS", True)
    USE_COMPLETION_TOKENS = (os.environ.get("AUGGIE_LAUNCH_USE_COMPLETION_TOKENS") or "auto").strip().lower()
    ENABLE_CONNECTION_POOL = env_truthy("AUGGIE_LAUNCH_CONNECTION_POOL", True)
    AUTO_INJECT_MCP = env_truthy("AUGGIE_LAUNCH_AUTO_INJECT_MCP", True)
    ENABLE_PLAN_MODE = env_truthy("AUGGIE_LAUNCH_PLAN_MODE", True)
    SESSION_MODE = (os.environ.get("AUGGIE_LAUNCH_MODE") or "").strip().lower()
    POST_RUN_CHECKS = env_truthy("AUGGIE_LAUNCH_POST_RUN_CHECKS", True)
    AGENT_MAX_ITERATIONS = env_int("AUGGIE_LAUNCH_AGENT_MAX_ITERATIONS", 200)
    FULL_ACCESS_MAX_ITERATIONS = env_int("AUGGIE_LAUNCH_FULL_ACCESS_MAX_ITERATIONS", 10000)
    ENABLE_PERSONA = env_truthy("AUGGIE_LAUNCH_PERSONA", True)

    if env_truthy("AUGGIE_LAUNCH_CONFIG_CHECK", True):
        report_config_warnings(validate_config())


def validate_config() -> list[str]:
    """Reports configuration that will not do what the user expects.

    Returns warnings rather than raising: a mistyped provider should not stop the
    launcher, but it must not be discovered halfway through a long turn either.
    """
    warnings: list[str] = []

    if IS_CODEGPT:
        if CODEGPT_PROVIDER:
            try:
                from . import codegpt
                # Read the catalog directly: provider_for_model() would just hand
                # back the pinned value and the comparison would always match.
                derived = ""
                wanted = (TARGET_MODEL or "").strip().lower()
                for entry in codegpt.load_catalog_models():
                    if entry["id"].lower() == wanted:
                        derived = str(entry.get("provider") or "")
                        break
                if derived and derived != CODEGPT_PROVIDER:
                    warnings.append(
                        f"AUGGIE_LAUNCH_CODEGPT_PROVIDER={CODEGPT_PROVIDER} is pinned, but "
                        f"{TARGET_MODEL} is served by {derived}; requests will likely fail"
                    )
            except Exception:
                pass
        if not CODEGPT_TOKEN and not CODEGPT_SESSION_URL:
            warnings.append(
                "no CodeGPT token and no session URL: set AUGGIE_LAUNCH_CODEGPT_TOKEN "
                "or let the VS Code extension serve one"
            )
    else:
        if not API_KEYS:
            warnings.append("AUGGIE_LAUNCH_API_KEY is empty")

    if TARGET_BASE_URL.startswith("http://") and "localhost" not in TARGET_BASE_URL and "127.0.0.1" not in TARGET_BASE_URL:
        warnings.append(f"AUGGIE_LAUNCH_BASE_URL={TARGET_BASE_URL} is plain http to a remote host")

    if MODEL_CONTEXT_TOKENS_EXPLICIT and MODEL_MAX_OUTPUT_TOKENS >= MODEL_CONTEXT_TOKENS:
        warnings.append(
            f"AUGGIE_LAUNCH_MODEL_MAX_OUTPUT_TOKENS ({MODEL_MAX_OUTPUT_TOKENS}) leaves no room "
            f"inside the {MODEL_CONTEXT_TOKENS}-token context"
        )

    return warnings


def report_config_warnings(warnings: list[str]) -> None:
    """Prints warnings to stderr, once, in a form that is easy to grep."""
    for warning in warnings:
        print(f"[auggie-launch] config warning: {warning}", file=sys.stderr)


def log(message: str) -> None:
    if VERBOSE:
        print(f"[auggie-launch] {message}", file=sys.stderr)


def detect_codegpt(base_url: str) -> bool:
    """True when the upstream is CodeGPT Plus, which routes chat through an agent."""
    if env_truthy("AUGGIE_LAUNCH_FORCE_CODEGPT", False):
        return True
    lowered = (base_url or "").lower()
    return "api.codegpt.co" in lowered or "codegpt.co/api" in lowered


