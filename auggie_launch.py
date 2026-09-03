#!/usr/bin/env python3
# Copyright (c) 2026 auggie-launch contributors
# For learning and research only; any other use is at your own risk.
"""auggie-launch: local Python proxy and launcher for Auggie/Augment Code CLI.

Deeply integrates with 9router (reading ~/.9router local database, combos, aliases,
catalog, tunnel, and provider keys), provides full injections into Auggie CLI
(session auth, environment, MCP tools, Caveman ultra instructions, feature flags),
and forwards chat/completion requests to OpenAI/Claude-compatible upstream endpoints.
"""

from __future__ import annotations

import http.client
import json
import os
import platform
import random
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
# Runtime upstream override: set when the local 9router socket dies and the
# Cloudflare tunnel takes over. Sticky for the process lifetime.
# lc-debt: no automatic switch back to local once the tunnel takes over; restart to reset.
ACTIVE_BASE_URL = ""
TUNNEL_BASE_URL = ""
BASE_URL_LOCK = threading.Lock()
TARGET_MODEL = ""
TARGET_API_KEY = ""
API_KEYS: list[str] = []
FROZEN_KEYS: dict[str, float] = {}
FROZEN_LOCK = threading.Lock()
AUGGIE_BIN = "auggie"
LOCAL_TOKEN = "fake-augment-access-token"
VERBOSE = False
DEBUG_DIR = ""
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
UPSTREAM_NEXT_REQUEST_AT = 0.0
UPSTREAM_COOLDOWN_UNTIL = 0.0
UPSTREAM_THROTTLE_LOCK = threading.Lock()

SANITIZE_UPSTREAM_PROMPTS = False
INDEXING_MODE = "complete"
MODEL_CONTEXT_TOKENS = 200000
MODEL_CONTEXT_TOKENS_EXPLICIT = False
MODEL_MAX_OUTPUT_TOKENS = 16000
REQUIRE_LOCAL_TOKEN = True
REASONING_EFFORT = ""
STREAM_THINKING = True

# --- 9router & Modern LLM Specific Features ---
IS_9ROUTER = False
ROUTER_CAVEMAN_MODE = False
ROUTER_CAVEMAN_LEVEL = "ultra"
ROUTER_PROVIDER = ""
CACHED_CATALOG: dict[str, Any] = {}
DYNAMIC_MODELS = True
USE_COMPLETION_TOKENS = "auto"  # auto | true | false
ENABLE_CONNECTION_POOL = True
AUTO_INJECT_MCP = True

_LOADED_ENV_FILES: list[str] = []
_CODEX_SESSION_ID = f"session-{uuid.uuid4()}"
_CODEX_THREAD_ID = str(uuid.uuid4())
_CODEX_INSTALLATION_ID = str(uuid.uuid4())

# Cached dynamic model registry from 9router/upstream
_CACHED_MODELS: list[dict[str, Any]] = []
_CACHED_MODELS_TIME = 0.0
_MODELS_CACHE_TTL = 60.0
_MODELS_LOCK = threading.Lock()


# ============================================================================
# Deep 9router Local State & Discovery Engine
# ============================================================================

@dataclass
class NineRouterLocalState:
    installed: bool = False
    db_path: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    combos: list[dict[str, Any]] = field(default_factory=list)
    model_aliases: dict[str, str] = field(default_factory=dict)
    api_keys: list[str] = field(default_factory=list)
    provider_connections: list[dict[str, Any]] = field(default_factory=list)
    provider_api_keys: dict[str, str] = field(default_factory=dict)
    tunnel_url: str = ""
    caveman_enabled: bool = False
    caveman_level: str = "ultra"
    catalog_models: dict[str, Any] = field(default_factory=dict)


def read_local_9router_state() -> NineRouterLocalState:
    """Reads ~/.9router/db.json and model-catalog.json to treat 9router as part of ourselves."""
    state = NineRouterLocalState()
    home = os.path.expanduser("~")
    nine_dir = os.path.join(home, ".9router")
    db_file = os.path.join(nine_dir, "db.json")
    catalog_file = os.path.join(nine_dir, "model-catalog.json")

    if not os.path.isdir(nine_dir) or not os.path.isfile(db_file):
        return state

    state.installed = True
    state.db_path = db_file

    try:
        with open(db_file, encoding="utf-8") as f:
            data = json.load(f)

        state.settings = data.get("settings") or {}
        state.combos = data.get("combos") or []
        state.model_aliases = data.get("modelAliases") or {}
        state.tunnel_url = (state.settings.get("tunnelUrl") or "").strip()
        state.caveman_enabled = bool(state.settings.get("cavemanEnabled", False))
        state.caveman_level = str(state.settings.get("cavemanLevel") or "ultra").strip()

        # Extract active API keys
        raw_keys = data.get("apiKeys") or []
        for k in raw_keys:
            if isinstance(k, dict) and k.get("key") and k.get("isActive", True):
                state.api_keys.append(str(k["key"]).strip())

        # Extract provider connections and API keys
        raw_providers = data.get("providerConnections") or []
        state.provider_connections = raw_providers
        for p in raw_providers:
            if not isinstance(p, dict) or not p.get("isActive", True):
                continue
            prov_name = str(p.get("provider") or "").lower()
            key_val = p.get("apiKey") or p.get("accessToken")
            if prov_name and key_val:
                state.provider_api_keys[prov_name] = str(key_val).strip()

    except Exception as exc:
        log(f"error reading local 9router db: {exc}")

    # Read model catalog for capabilities (vision, audio, pdf)
    if os.path.isfile(catalog_file):
        try:
            with open(catalog_file, encoding="utf-8") as f:
                cat_data = json.load(f)
                state.catalog_models = cat_data.get("models") or {}
        except Exception as exc:
            log(f"error reading local 9router catalog: {exc}")

    return state


_LOCAL_9ROUTER = read_local_9router_state()


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


def detect_9router(url: str) -> bool:
    """Checks whether the upstream URL is likely 9router."""
    if env_truthy("AUGGIE_LAUNCH_FORCE_9ROUTER", False):
        return True
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.port == 20128:
            return True
        if "9router" in parsed.netloc.lower() or "9router" in parsed.path.lower():
            return True
    except Exception:
        pass
    return False


def load_config() -> None:
    global TARGET_BASE_URL, TARGET_MODEL, TARGET_API_KEY, API_KEYS, AUGGIE_BIN
    global LOCAL_TOKEN, VERBOSE, DEBUG_DIR, PORT, _LOADED_ENV_FILES
    global UPSTREAM_USER_AGENT, UPSTREAM_APP_NAME, SANITIZE_UPSTREAM_PROMPTS
    global UPSTREAM_MIN_INTERVAL_SECONDS, UPSTREAM_RETRIES
    global UPSTREAM_429_FREEZE_SECONDS, UPSTREAM_5XX_FREEZE_SECONDS
    global UPSTREAM_MAX_RETRY_AFTER_SECONDS, UPSTREAM_BACKOFF_INITIAL_SECONDS
    global UPSTREAM_BACKOFF_MAX_SECONDS
    global INDEXING_MODE, MODEL_CONTEXT_TOKENS, MODEL_MAX_OUTPUT_TOKENS, REASONING_EFFORT
    global MODEL_CONTEXT_TOKENS_EXPLICIT, REQUIRE_LOCAL_TOKEN
    global STREAM_THINKING, IS_9ROUTER, ROUTER_CAVEMAN_MODE, ROUTER_CAVEMAN_LEVEL
    global ROUTER_PROVIDER, DYNAMIC_MODELS, USE_COMPLETION_TOKENS, ENABLE_CONNECTION_POOL
    global CACHED_CATALOG, _LOCAL_9ROUTER, AUTO_INJECT_MCP

    _LOADED_ENV_FILES = load_dotenv_files()
    _LOCAL_9ROUTER = read_local_9router_state()

    # Read model catalog for capabilities (vision, audio, pdf, context window)
    home = os.path.expanduser("~")
    catalog_file = os.path.join(home, ".9router", "model-catalog.json")
    if os.path.isfile(catalog_file):
        try:
            with open(catalog_file, encoding="utf-8") as f:
                cat_data = json.load(f)
                global CACHED_CATALOG
                CACHED_CATALOG = cat_data.get("models", {})
        except Exception as exc:
            log(f"error reading local 9router catalog: {exc}")

    # Zero-config auto-detection from 9router local DB if available
    base_url_env = os.environ.get("AUGGIE_LAUNCH_BASE_URL", "").strip()
    if not base_url_env:
        base_url_env = "http://localhost:20128/v1"
        os.environ["AUGGIE_LAUNCH_BASE_URL"] = base_url_env

    model_env = os.environ.get("AUGGIE_LAUNCH_MODEL", "").strip()
    if not model_env:
        model_env = _LOCAL_9ROUTER.combos[0]["name"] if _LOCAL_9ROUTER.combos else "free"
        os.environ["AUGGIE_LAUNCH_MODEL"] = model_env

    key_text = os.environ.get("AUGGIE_LAUNCH_API_KEYS") or os.environ.get("AUGGIE_LAUNCH_API_KEY") or ""
    if not key_text and _LOCAL_9ROUTER.api_keys:
        key_text = _LOCAL_9ROUTER.api_keys[0]
        os.environ["AUGGIE_LAUNCH_API_KEY"] = key_text

    API_KEYS = [key.strip() for key in key_text.split(",") if key.strip()]

    missing = [key for key in _REQUIRED_KEYS if not (os.environ.get(key) or "").strip()]
    if not API_KEYS:
        missing.append("AUGGIE_LAUNCH_API_KEY (or 9router active key)")
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
    # Per-session random token unless pinned: the proxy is loopback-only but still
    # refuses requests that do not carry the token it injected into Auggie.
    LOCAL_TOKEN = (os.environ.get("AUGGIE_LAUNCH_LOCAL_TOKEN") or f"al-{uuid.uuid4().hex}").strip()
    VERBOSE = env_truthy("AUGGIE_LAUNCH_VERBOSE")
    DEBUG_DIR = (os.environ.get("AUGGIE_LAUNCH_DEBUG_DIR") or os.path.join(tempfile.gettempdir(), "auggie-launch")).strip()
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
    SANITIZE_UPSTREAM_PROMPTS = env_truthy("AUGGIE_LAUNCH_SANITIZE_UPSTREAM_PROMPTS", False)
    INDEXING_MODE = (os.environ.get("AUGGIE_LAUNCH_INDEXING_MODE") or "complete").strip().lower()
    MODEL_CONTEXT_TOKENS = env_int("AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS", 200000)
    MODEL_CONTEXT_TOKENS_EXPLICIT = bool((os.environ.get("AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS") or "").strip())
    REQUIRE_LOCAL_TOKEN = env_truthy("AUGGIE_LAUNCH_REQUIRE_LOCAL_TOKEN", True)
    MODEL_MAX_OUTPUT_TOKENS = env_int("AUGGIE_LAUNCH_MODEL_MAX_OUTPUT_TOKENS", 16000)
    REASONING_EFFORT = (os.environ.get("AUGGIE_LAUNCH_REASONING_EFFORT") or "").strip().lower()
    if REASONING_EFFORT and REASONING_EFFORT not in {"low", "medium", "high"}:
        print("error: AUGGIE_LAUNCH_REASONING_EFFORT must be low, medium, or high", file=sys.stderr)
        sys.exit(2)

    STREAM_THINKING = env_truthy("AUGGIE_LAUNCH_STREAM_THINKING", True)
    IS_9ROUTER = detect_9router(TARGET_BASE_URL)

    # Auto-install 9router when the system cannot detect it at all
    if IS_9ROUTER and not _LOCAL_9ROUTER.installed and env_truthy("AUGGIE_LAUNCH_AUTO_INSTALL_9ROUTER", True):
        if ensure_9router_installed():
            _LOCAL_9ROUTER = read_local_9router_state()

    # Tunnel fallback target: 9router's Cloudflare URL + the local base path (e.g. /v1)
    global TUNNEL_BASE_URL, ACTIVE_BASE_URL
    ACTIVE_BASE_URL = ""
    TUNNEL_BASE_URL = ""
    if _LOCAL_9ROUTER.tunnel_url:
        base_path = urllib.parse.urlparse(TARGET_BASE_URL).path.rstrip("/")
        TUNNEL_BASE_URL = _LOCAL_9ROUTER.tunnel_url.rstrip("/") + base_path

    ROUTER_CAVEMAN_MODE = env_truthy("AUGGIE_LAUNCH_9ROUTER_CAVEMAN", _LOCAL_9ROUTER.caveman_enabled)
    ROUTER_CAVEMAN_LEVEL = (os.environ.get("AUGGIE_LAUNCH_9ROUTER_CAVEMAN_LEVEL") or _LOCAL_9ROUTER.caveman_level).strip().lower()
    ROUTER_PROVIDER = (os.environ.get("AUGGIE_LAUNCH_9ROUTER_PROVIDER") or "").strip()
    DYNAMIC_MODELS = env_truthy("AUGGIE_LAUNCH_DYNAMIC_MODELS", True)
    USE_COMPLETION_TOKENS = (os.environ.get("AUGGIE_LAUNCH_USE_COMPLETION_TOKENS") or "auto").strip().lower()
    ENABLE_CONNECTION_POOL = env_truthy("AUGGIE_LAUNCH_CONNECTION_POOL", True)
    AUTO_INJECT_MCP = env_truthy("AUGGIE_LAUNCH_AUTO_INJECT_MCP", True)


def log(message: str) -> None:
    if VERBOSE:
        print(f"[auggie-launch] {message}", file=sys.stderr)


# ============================================================================
# Token Estimation & Turn-Atomic Context Truncation
# ============================================================================

def estimate_tokens_heuristic(text: str) -> int:
    """Accurate heuristic token count for code and structured text (~3.5 chars/token)."""
    if not text:
        return 0
    length = len(text)
    return max(1, int(length / 3.5) + 1)


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimates tokens for a chat message including role and tool calls."""
    tokens = 4  # Overhead for message wrapper
    content = msg.get("content")
    if isinstance(content, str):
        tokens += estimate_tokens_heuristic(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                tokens += estimate_tokens_heuristic(str(part.get("text") or part.get("content") or ""))
            elif isinstance(part, str):
                tokens += estimate_tokens_heuristic(part)

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                tokens += estimate_tokens_heuristic(str(fn.get("name") or ""))
                tokens += estimate_tokens_heuristic(str(fn.get("arguments") or ""))
                tokens += 8
    return tokens


@dataclass
class ConversationTurn:
    messages: list[dict[str, Any]]
    estimated_tokens: int


def group_messages_into_turns(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[ConversationTurn]]:
    """Groups conversation messages into atomic turns.
    CRITICAL: Never separates an assistant tool_call from its corresponding tool results!
    OpenAI and Anthropic APIs return 400 Bad Request if an assistant message with
    'tool_calls' is not immediately followed by tool messages responding to each tool_call_id.
    """
    system_messages: list[dict[str, Any]] = []
    turns: list[ConversationTurn] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = str(msg.get("role") or "").lower()

        if role == "system":
            system_messages.append(msg)
            i += 1
            continue

        if role == "assistant" and msg.get("tool_calls"):
            turn_msgs = [msg]
            j = i + 1
            while j < len(messages) and str(messages[j].get("role") or "").lower() == "tool":
                turn_msgs.append(messages[j])
                j += 1
            i = j
            turn_tokens = sum(estimate_message_tokens(m) for m in turn_msgs)
            turns.append(ConversationTurn(messages=turn_msgs, estimated_tokens=turn_tokens))
        else:
            turn_msgs = [msg]
            turn_tokens = estimate_message_tokens(msg)
            turns.append(ConversationTurn(messages=turn_msgs, estimated_tokens=turn_tokens))
            i += 1

    return system_messages, turns


def truncate_messages_to_context_limit(messages: list[dict[str, Any]], max_context_tokens: int) -> list[dict[str, Any]]:
    """Prunes messages while guaranteeing turn-atomic integrity and tool_call pairing."""
    if not messages or max_context_tokens <= 0:
        return messages

    reserve = max(MODEL_MAX_OUTPUT_TOKENS, 4096) + 1024
    available_budget = max(4096, max_context_tokens - reserve)

    system_msgs, turns = group_messages_into_turns(messages)
    system_tokens = sum(estimate_message_tokens(m) for m in system_msgs)
    total_tokens = system_tokens + sum(t.estimated_tokens for t in turns)

    if total_tokens <= available_budget:
        return messages

    log(f"context tokens ({total_tokens}) exceeds budget ({available_budget}), performing turn-atomic truncation...")

    if not turns:
        return system_msgs

    latest_turn = turns[-1]
    remaining_budget = available_budget - system_tokens - latest_turn.estimated_tokens

    first_turn = turns[0] if len(turns) > 1 else None
    middle_turns = turns[1:-1] if len(turns) > 2 else []

    if first_turn and remaining_budget >= first_turn.estimated_tokens:
        kept_first = first_turn
        remaining_budget -= first_turn.estimated_tokens
    else:
        kept_first = None

    kept_middle: list[ConversationTurn] = []
    for turn in reversed(middle_turns):
        if remaining_budget >= turn.estimated_tokens:
            kept_middle.append(turn)
            remaining_budget -= turn.estimated_tokens
        else:
            if len(turn.messages) == 1 and turn.messages[0].get("role") in {"user", "tool"}:
                single_msg = dict(turn.messages[0])
                content = str(single_msg.get("content") or "")
                char_budget = max(200, int(remaining_budget * 3.0))
                if len(content) > char_budget:
                    single_msg["content"] = content[:char_budget] + "\n... [truncated due to context limit]"
                    trimmed_turn = ConversationTurn(messages=[single_msg], estimated_tokens=estimate_message_tokens(single_msg))
                    kept_middle.append(trimmed_turn)
            break

    kept_middle.reverse()

    result_turns: list[ConversationTurn] = []
    if kept_first:
        result_turns.append(kept_first)
    result_turns.extend(kept_middle)
    result_turns.append(latest_turn)

    out: list[dict[str, Any]] = list(system_msgs)
    for t in result_turns:
        out.extend(t.messages)
    return out


# ============================================================================
# Tool Calls Merging & JSON Repair
# ============================================================================

def repair_json_arguments(raw_args: str) -> str:
    """Attempts to repair truncated or unclosed JSON arguments from streaming chunks."""
    s = raw_args.strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    repaired = s
    if repaired.count('"') % 2 != 0:
        repaired += '"'
    open_curly = repaired.count("{") - repaired.count("}")
    open_square = repaired.count("[") - repaired.count("]")
    repaired += ("]" * max(0, open_square)) + ("}" * max(0, open_curly))
    try:
        json.loads(repaired)
        return repaired
    except Exception:
        return s


def merge_stream_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically merges streaming delta tool calls by index and id."""
    calls_by_index: dict[int, dict[str, Any]] = {}

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        idx = call.get("index")
        if idx is None:
            idx = len(calls_by_index)
        if idx not in calls_by_index:
            calls_by_index[idx] = {
                "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }
        target = calls_by_index[idx]
        if call.get("id"):
            target["id"] = call["id"]

        fn = call.get("function")
        if isinstance(fn, dict):
            if fn.get("name"):
                target["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                target["function"]["arguments"] += fn["arguments"]

    result: list[dict[str, Any]] = []
    for idx in sorted(calls_by_index.keys()):
        item = calls_by_index[idx]
        name = item["function"]["name"].strip()
        if not name:
            continue
        item["function"]["arguments"] = repair_json_arguments(item["function"]["arguments"])
        result.append(item)
    return result


def tool_calls_to_nodes(tool_calls: list[dict[str, Any]], *, starting_id: int = 2) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    node_id = starting_id
    for call in merge_stream_tool_calls(tool_calls):
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        fn = fn or {}
        name = fn.get("name")
        args = fn.get("arguments") if isinstance(fn.get("arguments"), str) else "{}"
        call_id = call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        if not isinstance(name, str) or not name:
            continue
        nodes.append({"id": node_id, "type": 5, "tool_use": {"tool_name": name, "tool_use_id": str(call_id), "input_json": args}})
        node_id += 1
    return nodes


# ============================================================================
# OpenAI Message & Tool Transformation
# ============================================================================

def text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                parts.append(entry)
            elif isinstance(entry, dict):
                text = entry.get("text") or entry.get("content") or entry.get("message")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "message", "prompt", "query", "input"):
            if isinstance(value.get(key), str):
                return str(value[key])
    return ""


def normalize_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_schema(val.lower() if key == "type" and isinstance(val, str) else val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    return value


def parse_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    for key in ("input_schema_json", "input_schema", "parameters"):
        schema = tool.get(key)
        if isinstance(schema, dict):
            return normalize_schema(schema)
        if isinstance(schema, str) and schema.strip():
            try:
                parsed = json.loads(schema)
                if isinstance(parsed, dict):
                    return normalize_schema(parsed)
            except Exception:
                pass
    return {"type": "object", "properties": {}}


def build_openai_tools(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    definitions = body.get("tool_definitions")
    if not isinstance(definitions, list):
        return []
    tools: list[dict[str, Any]] = []
    for item in definitions:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(item.get("description") or ""),
                "parameters": parse_tool_schema(item),
            },
        })
    return tools


def node_text(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    text_node = node.get("text_node")
    if isinstance(text_node, dict):
        text = text_from_value(text_node.get("content") or text_node.get("text"))
        if text:
            return text
    return text_from_value(node.get("content") or node.get("text") or node.get("message"))


def node_tool_use(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    tool_use = node.get("tool_use")
    if isinstance(tool_use, dict):
        name = tool_use.get("tool_name") or tool_use.get("name")
        call_id = tool_use.get("tool_use_id") or tool_use.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        args = tool_use.get("input_json") or tool_use.get("arguments") or tool_use.get("input") or "{}"
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        if isinstance(name, str) and name:
            return {"id": str(call_id), "type": "function", "function": {"name": name, "arguments": args}}
    return None


def node_tool_result(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    result = node.get("tool_result_node") or node.get("tool_result") or node.get("tool_use_result") or node.get("toolUseResult")
    if isinstance(result, dict):
        call_id = result.get("tool_use_id") or result.get("tool_call_id") or result.get("id")
        content = result.get("content") or result.get("result") or result.get("output") or result.get("text") or result
        if call_id:
            return {"role": "tool", "tool_call_id": str(call_id), "content": text_from_value(content) or json.dumps(content, ensure_ascii=False)}
    return None


def append_history_messages(messages: list[dict[str, Any]], body: Any) -> None:
    if not isinstance(body, dict):
        return
    history = body.get("chat_history") or body.get("history")
    if not isinstance(history, list):
        return
    for record in history:
        if not isinstance(record, dict):
            continue
        request_nodes = record.get("request_nodes") if isinstance(record.get("request_nodes"), list) else []
        response_nodes = record.get("response_nodes") if isinstance(record.get("response_nodes"), list) else []
        request_nodes = request_nodes or []
        response_nodes = response_nodes or []
        for node in request_nodes:
            tool_result = node_tool_result(node)
            if tool_result:
                messages.append(tool_result)
        user_text = "\n".join(filter(None, (node_text(node) for node in request_nodes))) or text_from_value(record.get("request_message"))
        if user_text:
            messages.append({"role": "user", "content": user_text})
        assistant_text = "\n".join(filter(None, (node_text(node) for node in response_nodes))) or text_from_value(record.get("response_text"))
        tool_calls = [call for call in (node_tool_use(node) for node in response_nodes) if call]
        if assistant_text or tool_calls:
            msg: dict[str, Any] = {"role": "assistant", "content": assistant_text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)


def append_current_tool_results(messages: list[dict[str, Any]], body: Any) -> None:
    if not isinstance(body, dict):
        return
    nodes = body.get("nodes")
    if not isinstance(nodes, list):
        return
    for node in nodes:
        tool_result = node_tool_result(node)
        if tool_result:
            messages.append(tool_result)


def current_message_text(body: Any) -> str:
    if not isinstance(body, dict):
        return text_from_value(body)
    for key in ("message", "request_message", "current_message", "prompt", "query", "input"):
        value = text_from_value(body.get(key)).strip()
        if value:
            return value
    request_nodes = body.get("request_nodes")
    if isinstance(request_nodes, list):
        text = "\n".join(filter(None, (node_text(node) for node in request_nodes))).strip()
        if text:
            return text
    nodes = body.get("nodes")
    if isinstance(nodes, list):
        text = "\n".join(filter(None, (node_text(node) for node in nodes))).strip()
        if text:
            return text
    return ""


def build_system_prompt() -> str:
    """Combines custom system prompt with 9router Caveman mode instructions if enabled."""
    parts: list[str] = []
    base_prompt = os.environ.get("AUGGIE_LAUNCH_SYSTEM_PROMPT", "").strip()
    if base_prompt:
        parts.append(base_prompt)

    if ROUTER_CAVEMAN_MODE:
        if ROUTER_CAVEMAN_LEVEL == "ultra":
            parts.append("Respond with maximum brevity. Output only working code and essential commands. Omit pleasantries, conversational intro/outro, and obvious explanations.")
        elif ROUTER_CAVEMAN_LEVEL == "lite":
            parts.append("Be concise and direct. Keep code explanations brief.")
        else:
            parts.append("Keep answers brief, code-focused, and eliminate conversational filler.")

    return "\n\n".join(parts)


def build_openai_messages(body: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = build_system_prompt()
    if system:
        messages.append({"role": "system", "content": system})

    append_history_messages(messages, body)
    append_current_tool_results(messages, body)

    if isinstance(body, dict):
        raw_messages = body.get("messages")
        if isinstance(raw_messages, list):
            for entry in raw_messages:
                if not isinstance(entry, dict):
                    continue
                role = str(entry.get("role") or entry.get("speaker") or "user").lower()
                if role in {"assistant", "agent", "bot"}:
                    out_role = "assistant"
                elif role in {"system", "developer"}:
                    out_role = "system"
                else:
                    out_role = "user"
                content = text_from_value(entry.get("content") or entry.get("text") or entry.get("message"))
                if content:
                    messages.append({"role": out_role, "content": content})

    current = current_message_text(body)
    if current:
        if not messages or messages[-1].get("content") != current:
            messages.append({"role": "user", "content": current})
    if not messages:
        messages.append({"role": "user", "content": "(empty request)"})
    return messages


def should_use_completion_tokens(model_name: str) -> bool:
    """Detects whether model requires max_completion_tokens instead of max_tokens."""
    if USE_COMPLETION_TOKENS == "true":
        return True
    if USE_COMPLETION_TOKENS == "false":
        return False
    lower = model_name.lower()
    if any(prefix in lower for prefix in ("o1-", "o1", "o3-", "o3", "claude-3-7", "deepseek-r1")):
        return True
    return False


def build_openai_request(body: Any, *, stream: bool, use_completion_tokens: bool | None = None) -> dict[str, Any]:
    raw_messages = build_openai_messages(body)
    model = resolve_request_model(body)
    limit = effective_context_limit(model)
    if limit <= 0:
        limit = 200000
    messages = truncate_messages_to_context_limit(raw_messages, limit)

    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    tools = build_openai_tools(body)
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"
    if stream:
        request["stream_options"] = {"include_usage": True}

    tokens_field = "max_completion_tokens" if (
        use_completion_tokens if use_completion_tokens is not None else should_use_completion_tokens(model)
    ) else "max_tokens"

    # Never ask for more output than the model's window can hold alongside the prompt.
    output_ceiling = max(256, min(MODEL_MAX_OUTPUT_TOKENS, max(1024, limit // 4)))

    if isinstance(body, dict):
        if isinstance(body.get("temperature"), (int, float)):
            request["temperature"] = body["temperature"]
        max_tokens = body.get("max_tokens") or body.get("max_output_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            request[tokens_field] = min(max_tokens, output_ceiling)
        if isinstance(body.get("reasoning_effort"), str) and body["reasoning_effort"].strip():
            request["reasoning_effort"] = body["reasoning_effort"].strip().lower()

    if "reasoning_effort" not in request and REASONING_EFFORT:
        request["reasoning_effort"] = REASONING_EFFORT

    return request


# ============================================================================
# HTTP Connection Pooling & Networking
# ============================================================================

class ConnectionPool:
    """Thread-safe HTTP/1.1 persistent connection pool with Keep-Alive."""

    def __init__(self, max_idle_seconds: float = 60.0, pool_size_per_host: int = 8):
        self._connections: dict[str, list[tuple[http.client.HTTPConnection, float]]] = {}
        self._lock = threading.Lock()
        self._max_idle = max_idle_seconds
        self._pool_size = pool_size_per_host

    def acquire(self, parsed_url: urllib.parse.ParseResult, timeout: float = 300.0) -> http.client.HTTPConnection:
        key = f"{parsed_url.scheme}://{parsed_url.netloc}"
        now = time.time()
        with self._lock:
            pool = self._connections.setdefault(key, [])
            while pool:
                conn, last_used = pool.pop()
                if (now - last_used) <= self._max_idle and conn.sock is not None:
                    return conn
                try:
                    conn.close()
                except Exception:
                    pass

        is_https = parsed_url.scheme == "https"
        host = parsed_url.hostname or "127.0.0.1"
        port = parsed_url.port or (443 if is_https else 80)
        if is_https:
            import ssl
            context = ssl.create_default_context()
            return http.client.HTTPSConnection(host, port, timeout=timeout, context=context)
        else:
            return http.client.HTTPConnection(host, port, timeout=timeout)

    def release(self, parsed_url: urllib.parse.ParseResult, conn: http.client.HTTPConnection, reusable: bool = True) -> None:
        key = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if not reusable or conn.sock is None:
            try:
                conn.close()
            except Exception:
                pass
            return
        with self._lock:
            pool = self._connections.setdefault(key, [])
            if len(pool) < self._pool_size:
                pool.append((conn, time.time()))
            else:
                try:
                    conn.close()
                except Exception:
                    pass


_CONNECTION_POOL = ConnectionPool()


def upstream_headers(api_key: str, *, stream: bool) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "Connection": "keep-alive",
    }
    if UPSTREAM_USER_AGENT:
        headers["User-Agent"] = UPSTREAM_USER_AGENT

    # 9router native headers
    if IS_9ROUTER or env_truthy("AUGGIE_LAUNCH_FORCE_9ROUTER", False):
        headers["X-Source"] = "auggie-launch"
        headers["X-RTK"] = "true"  # Enable Rust Token Killer compression
        if ROUTER_CAVEMAN_MODE:
            headers["X-Caveman-Mode"] = "true"
            headers["X-Caveman-Level"] = ROUTER_CAVEMAN_LEVEL
        if ROUTER_PROVIDER:
            headers["X-Router-Provider"] = ROUTER_PROVIDER

    return with_codex_headers(headers)


def active_base_url() -> str:
    return ACTIVE_BASE_URL or TARGET_BASE_URL


def switch_to_tunnel(reason: str) -> bool:
    """Fails the live upstream over to the 9router Cloudflare tunnel. True if switched."""
    global ACTIVE_BASE_URL
    if not TUNNEL_BASE_URL:
        return False
    with BASE_URL_LOCK:
        if active_base_url() == TUNNEL_BASE_URL:
            return False
        if not _ping_tunnel(TUNNEL_BASE_URL):
            return False
        ACTIVE_BASE_URL = TUNNEL_BASE_URL
    print(
        f"[auggie-launch] local 9router unreachable ({reason}); failing over to tunnel {TUNNEL_BASE_URL}",
        file=sys.stderr,
    )
    return True


def upstream_url() -> str:
    return f"{active_base_url()}/chat/completions"


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt is None:
            return None
        return max(0.0, dt.timestamp() - (time.time() if now is None else now))
    except Exception:
        return None


def clamp_retry_delay(delay: float | None, fallback: float) -> float:
    selected = fallback if delay is None else delay
    return min(max(0.0, selected), UPSTREAM_MAX_RETRY_AFTER_SECONDS)


def retry_backoff_seconds(attempt: int) -> float:
    """Full jitter exponential backoff (AWS architecture standard)."""
    base = min(UPSTREAM_BACKOFF_MAX_SECONDS, UPSTREAM_BACKOFF_INITIAL_SECONDS * (2 ** max(0, attempt)))
    return random.uniform(0.0, base)


def apply_upstream_cooldown(seconds: float, reason: str) -> None:
    if seconds <= 0:
        return
    capped = min(seconds, UPSTREAM_MAX_RETRY_AFTER_SECONDS)
    with UPSTREAM_THROTTLE_LOCK:
        global UPSTREAM_COOLDOWN_UNTIL
        until = time.time() + capped
        if until > UPSTREAM_COOLDOWN_UNTIL:
            UPSTREAM_COOLDOWN_UNTIL = until
    log(f"upstream cooldown {capped:.2f}s ({reason})")


def wait_for_upstream_slot() -> None:
    global UPSTREAM_NEXT_REQUEST_AT
    while True:
        with UPSTREAM_THROTTLE_LOCK:
            now = time.time()
            wait_until = max(UPSTREAM_COOLDOWN_UNTIL, UPSTREAM_NEXT_REQUEST_AT)
            wait_for = wait_until - now
            if wait_for <= 0:
                UPSTREAM_NEXT_REQUEST_AT = now + UPSTREAM_MIN_INTERVAL_SECONDS
                return
        time.sleep(min(wait_for, 5.0))


def reset_throttles() -> None:
    """Resets all throttling and cooldown timers."""
    global UPSTREAM_NEXT_REQUEST_AT, UPSTREAM_COOLDOWN_UNTIL
    with UPSTREAM_THROTTLE_LOCK:
        UPSTREAM_NEXT_REQUEST_AT = 0.0
        UPSTREAM_COOLDOWN_UNTIL = 0.0


def is_retryable_upstream_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def compact_upstream_error(message: str, *, max_chars: int = 1000) -> str:
    text = (message or "").strip()
    lower = text.lower()
    if "<html" in lower or "<!doctype html" in lower:
        title = ""
        title_start = lower.find("<title>")
        title_end = lower.find("</title>")
        if 0 <= title_start < title_end:
            title = text[title_start + len("<title>"):title_end].strip()
        text = f"HTML upstream error page: {title}" if title else "HTML upstream error page returned by gateway"
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def get_active_key() -> str:
    with FROZEN_LOCK:
        now = time.time()
        for key in API_KEYS:
            if FROZEN_KEYS.get(key, 0) <= now:
                return key
        return API_KEYS[0] if API_KEYS else ""


def mark_key_failed(key: str, status_code: int, retry_after: str | None = None) -> float:
    with FROZEN_LOCK:
        now = time.time()
        if status_code == 429:
            freeze_for = clamp_retry_delay(parse_retry_after(retry_after, now=now), UPSTREAM_429_FREEZE_SECONDS)
            FROZEN_KEYS[key] = now + freeze_for
            return freeze_for
        elif status_code in (401, 402):
            FROZEN_KEYS[key] = now + 86400
            return 86400.0
        elif 500 <= status_code <= 599:
            freeze_for = UPSTREAM_5XX_FREEZE_SECONDS
            FROZEN_KEYS[key] = now + freeze_for
            return freeze_for
    return 0.0


def log_router_response_headers(headers: dict[str, str] | http.client.HTTPMessage) -> None:
    """Logs 9router fallback, provider, and rate-limit metadata in verbose mode."""
    if not VERBOSE:
        return
    items = dict(headers.items()) if hasattr(headers, "items") else dict(headers)
    router_provider = items.get("x-router-provider") or items.get("X-Router-Provider")
    model_used = items.get("x-model-used") or items.get("X-Model-Used")
    token_savings = items.get("x-token-savings") or items.get("X-Token-Savings")
    rate_remaining = items.get("x-ratelimit-remaining-requests") or items.get("X-RateLimit-Remaining-Requests")

    details = []
    if router_provider:
        details.append(f"provider={router_provider}")
    if model_used:
        details.append(f"model={model_used}")
    if token_savings:
        details.append(f"savings={token_savings}")
    if rate_remaining:
        details.append(f"rate_rem={rate_remaining}")
    if details:
        log(f"9router: {' | '.join(details)}")


class UpstreamResponseWrapper:
    """Wrapper that ensures connection release back to the pool."""

    def __init__(self, parsed_url: urllib.parse.ParseResult, conn: http.client.HTTPConnection, response: http.client.HTTPResponse):
        self.parsed_url = parsed_url
        self.conn = conn
        self.response = response
        self.status = getattr(response, "status", 200)
        self.headers = getattr(response, "headers", {})
        self._closed = False

    def read(self, amt: int | None = None) -> bytes:
        return self.response.read(amt)

    def readline(self) -> bytes:
        return self.response.readline()

    def __iter__(self):
        return iter(self.response)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            reusable = getattr(self.response, "isclosed", lambda: False)
            is_closed = reusable() if callable(reusable) else bool(reusable)
            _CONNECTION_POOL.release(self.parsed_url, self.conn, reusable=not is_closed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def open_upstream_with_retries(data: bytes, *, stream: bool, timeout: int, label: str) -> UpstreamResponseWrapper:
    """Executes upstream HTTP requests with Keep-Alive pool, full jitter, and parameter negotiation."""
    parsed_url = urllib.parse.urlparse(upstream_url())
    path_with_query = parsed_url.path or "/chat/completions"
    if parsed_url.query:
        path_with_query += f"?{parsed_url.query}"

    last_error: Exception | None = None
    max_attempts = max(len(API_KEYS), 1) + max(0, UPSTREAM_RETRIES)
    current_data = data

    for attempt in range(max_attempts):
        api_key = get_active_key()
        log(f"{label} upstream attempt={attempt + 1}/{max_attempts} key={api_key[:10]}...")

        headers = upstream_headers(api_key, stream=stream)
        conn = _CONNECTION_POOL.acquire(parsed_url, timeout=timeout)
        wait_for_upstream_slot()

        try:
            conn.request("POST", path_with_query, body=current_data, headers=headers)
            response = conn.getresponse()
            log_router_response_headers(response.headers)

            if response.status < 400:
                return UpstreamResponseWrapper(parsed_url, conn, response)

            error_body = response.read().decode("utf-8", errors="replace")
            retry_after = response.headers.get("Retry-After")
            frozen_for = mark_key_failed(api_key, response.status, retry_after)

            # Check for parameter incompatibility error (max_tokens vs max_completion_tokens)
            if response.status == 400 and ("max_tokens" in error_body or "max_completion_tokens" in error_body):
                try:
                    payload = json.loads(current_data.decode("utf-8"))
                    if "max_tokens" in payload:
                        payload["max_completion_tokens"] = payload.pop("max_tokens")
                        current_data = json_bytes(payload)
                        log("auto-swapped max_tokens to max_completion_tokens on 400 error, retrying immediately...")
                        _CONNECTION_POOL.release(parsed_url, conn, reusable=False)
                        continue
                    elif "max_completion_tokens" in payload:
                        payload["max_tokens"] = payload.pop("max_completion_tokens")
                        current_data = json_bytes(payload)
                        log("auto-swapped max_completion_tokens to max_tokens on 400 error, retrying immediately...")
                        _CONNECTION_POOL.release(parsed_url, conn, reusable=False)
                        continue
                except Exception:
                    pass

            last_error = urllib.error.HTTPError(
                upstream_url(), response.status, compact_upstream_error(error_body), response.headers, None
            )

            if response.status == 429:
                cooldown = clamp_retry_delay(
                    parse_retry_after(retry_after),
                    min(frozen_for or UPSTREAM_429_FREEZE_SECONDS, retry_backoff_seconds(attempt)),
                )
                apply_upstream_cooldown(cooldown, "429 rate limit")
            elif 500 <= response.status <= 599:
                apply_upstream_cooldown(retry_backoff_seconds(attempt), f"{response.status} upstream error")

            _CONNECTION_POOL.release(parsed_url, conn, reusable=False)

            if is_retryable_upstream_status(response.status) and attempt < max_attempts - 1:
                continue
            if response.status in (401, 402) and attempt < min(len(API_KEYS), max_attempts) - 1:
                continue
            break

        except Exception as exc:
            last_error = exc
            _CONNECTION_POOL.release(parsed_url, conn, reusable=False)
            if switch_to_tunnel(str(exc) or exc.__class__.__name__):
                parsed_url = urllib.parse.urlparse(upstream_url())
                path_with_query = parsed_url.path or "/chat/completions"
                if parsed_url.query:
                    path_with_query += f"?{parsed_url.query}"
                continue
            if attempt < max_attempts - 1:
                apply_upstream_cooldown(retry_backoff_seconds(attempt), "transport error")
                continue
            break

    if last_error is not None:
        raise last_error
    raise RuntimeError("upstream request failed")


# ============================================================================
# Dynamic Model Registry & 9router Catalog Discovery
# ============================================================================

def fetch_upstream_models() -> list[dict[str, Any]]:
    """Fetches real model list from 9router / upstream /v1/models endpoint."""
    global _CACHED_MODELS, _CACHED_MODELS_TIME
    now = time.time()
    with _MODELS_LOCK:
        if _CACHED_MODELS and (now - _CACHED_MODELS_TIME) < _MODELS_CACHE_TTL:
            return _CACHED_MODELS

    models_url = f"{active_base_url()}/models"
    parsed_url = urllib.parse.urlparse(models_url)
    headers = upstream_headers(get_active_key(), stream=False)
    conn = _CONNECTION_POOL.acquire(parsed_url, timeout=5.0)

    try:
        path = parsed_url.path or "/models"
        conn.request("GET", path, headers=headers)
        res = conn.getresponse()
        if res.status == 200:
            data = json.loads(res.read().decode("utf-8"))
            raw_list = data.get("data") if isinstance(data, dict) else []
            parsed_models: list[dict[str, Any]] = []
            if isinstance(raw_list, list):
                for item in raw_list:
                    if isinstance(item, dict) and item.get("id"):
                        parsed_models.append(item)
            if parsed_models:
                with _MODELS_LOCK:
                    _CACHED_MODELS = parsed_models
                    _CACHED_MODELS_TIME = now
                log(f"loaded {len(parsed_models)} dynamic models from upstream registry")
                _CONNECTION_POOL.release(parsed_url, conn, reusable=True)
                return parsed_models
    except Exception as exc:
        log(f"could not fetch dynamic models: {exc}")
    finally:
        _CONNECTION_POOL.release(parsed_url, conn, reusable=False)

    return _CACHED_MODELS


def lookup_catalog_context(model_id: str) -> int:
    """Get context window from 9router model-catalog.json, fallback to heuristic."""
    global CACHED_CATALOG
    if not CACHED_CATALOG:
        return 0
    # Exact model ID match
    if model_id in CACHED_CATALOG:
        entry = CACHED_CATALOG.get(model_id)
        if isinstance(entry, dict):
            ctx = entry.get("contextWindow") or entry.get("context_window") or entry.get("contextLength")
            if isinstance(ctx, (int, float)) and ctx > 0:
                return int(ctx)
    # Last segment match (e.g., "gpt-4o" from "openai/gpt-4o")
    if "/" in model_id:
        last_segment = model_id.split("/")[-1]
        if last_segment in CACHED_CATALOG:
            entry = CACHED_CATALOG.get(last_segment)
            if isinstance(entry, dict):
                ctx = entry.get("contextWindow") or entry.get("context_window") or entry.get("contextLength")
                if isinstance(ctx, (int, float)) and ctx > 0:
                    return int(ctx)
    return 0

def combo_context_limit(combo: dict[str, Any]) -> int:
    """Safe context for a 9router combo: the smallest window across its fallback models.

    A combo can be served by any member, so the usable context is the weakest one;
    anything larger would 400 as soon as the combo falls back.
    """
    members = [m for m in (combo.get("models") or []) if isinstance(m, str)]
    limits = [model_context_limit(m) for m in members]
    limits = [limit for limit in limits if limit > 0]
    if not limits:
        return MODEL_CONTEXT_TOKENS if MODEL_CONTEXT_TOKENS > 0 else 200000
    return min(limits)


def effective_context_limit(model_id: str) -> int:
    """Context budget actually used for truncation and injection for one model.

    An explicit AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS always wins (operator override);
    otherwise combos use their weakest member and everything else uses the catalog.
    """
    if MODEL_CONTEXT_TOKENS_EXPLICIT and MODEL_CONTEXT_TOKENS > 0:
        return MODEL_CONTEXT_TOKENS
    for combo in _LOCAL_9ROUTER.combos:
        if combo.get("name") == model_id:
            return combo_context_limit(combo)
    alias_target = _LOCAL_9ROUTER.model_aliases.get(model_id)
    if alias_target:
        aliased = model_context_limit(alias_target)
        if aliased > 0:
            return aliased
    return model_context_limit(model_id)


def known_model_names() -> set[str]:
    """Every model name the launcher advertises to Auggie."""
    names = {TARGET_MODEL}
    names.update(str(c.get("name")) for c in _LOCAL_9ROUTER.combos if c.get("name"))
    names.update(_LOCAL_9ROUTER.model_aliases.keys())
    names.update(
        str(item.get("id")) for item in _CACHED_MODELS if isinstance(item, dict) and item.get("id")
    )
    return names


def resolve_request_model(body: Any) -> str:
    """Honours the model Auggie asked for when we advertise it, else the configured target."""
    if not isinstance(body, dict):
        return TARGET_MODEL
    for key in ("model", "model_name", "internal_name", "modelName"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            if candidate in known_model_names():
                return candidate
            log(f"ignoring unknown requested model '{candidate}', using {TARGET_MODEL}")
            break
    return TARGET_MODEL


def model_context_limit(model_id: str) -> int:
    """Heuristic context window for standard and 9router models, with catalog priority."""
    catalog_ctx = lookup_catalog_context(model_id)
    if catalog_ctx > 0:
        return catalog_ctx
    lower = model_id.lower()
    if "gemini" in lower:
        return 1000000
    if "claude" in lower or "gpt-4o" in lower or "o1" in lower or "o3" in lower:
        return 200000
    if "deepseek" in lower or "qwen" in lower:
        return 128000
    return MODEL_CONTEXT_TOKENS


# ============================================================================
# Full Model Registry & Session Injection
# ============================================================================

def fake_token() -> dict[str, Any]:
    return {
        "access_token": LOCAL_TOKEN,
        "token_type": "Bearer",
        "expires_in": 31536000,
        "scope": "email profile offline_access",
    }


def model_list_entry(model_id: str, context_tokens: int) -> dict[str, Any]:
    """Auggie model descriptor whose completion budgets track the real context window.

    Auggie sizes its prefix/suffix payloads from these counts, so deriving them from
    the model's own window is what keeps large-context models usable and small ones
    from overflowing upstream.
    """
    context = context_tokens if context_tokens > 0 else 200000
    # ~4 chars/token: one quarter of the window per side leaves half the window
    # for history, tool definitions and the response.
    half_budget_chars = max(2000, min(200000, context))
    return {
        "name": model_id,
        "internal_name": model_id,
        "suggested_prefix_char_count": half_budget_chars,
        "suggested_suffix_char_count": half_budget_chars,
        "completion_timeout_ms": 600000,
    }


def fake_models() -> dict[str, Any]:
    """Builds comprehensive model list including all 9router combos, aliases, and catalog."""
    target_context = effective_context_limit(TARGET_MODEL)
    models_list = [model_list_entry(TARGET_MODEL, target_context)]
    model_registry: dict[str, Any] = {
        TARGET_MODEL: {
            "humanName": TARGET_MODEL,
            "description": f"{'9router' if IS_9ROUTER else 'OpenAI-compatible'} model via auggie-launch",
            "encoding": "o200k_base",
            "context": target_context,
            "maxOutput": MODEL_MAX_OUTPUT_TOKENS,
        }
    }

    seen_models: set[str] = {TARGET_MODEL}

    # 1. Inject 9router combos from local db.json
    for combo in _LOCAL_9ROUTER.combos:
        cname = combo.get("name")
        if not cname or cname in seen_models:
            continue
        seen_models.add(cname)
        cmodels = combo.get("models") or []
        combo_context = combo_context_limit(combo)
        models_list.append(model_list_entry(cname, combo_context))
        model_registry[cname] = {
            "humanName": f"9router: {cname} (Combo)",
            "description": f"Auto-fallback combo over {len(cmodels)} models: {', '.join(cmodels[:3])}...",
            "encoding": "o200k_base",
            "context": combo_context,
            "maxOutput": MODEL_MAX_OUTPUT_TOKENS,
        }

    # 2. Inject 9router model aliases from local db.json
    for alias_name, real_target in _LOCAL_9ROUTER.model_aliases.items():
        if not alias_name or alias_name in seen_models:
            continue
        seen_models.add(alias_name)
        alias_context = effective_context_limit(alias_name)
        models_list.append(model_list_entry(alias_name, alias_context))
        model_registry[alias_name] = {
            "humanName": f"9router: {alias_name}",
            "description": f"Alias pointing to {real_target}",
            "encoding": "o200k_base",
            "context": alias_context,
            "maxOutput": MODEL_MAX_OUTPUT_TOKENS,
        }

    # 3. Inject live models from upstream /v1/models if enabled
    if DYNAMIC_MODELS:
        dynamic_list = fetch_upstream_models()
        for item in dynamic_list:
            mid = item.get("id")
            if not mid or mid in seen_models:
                continue
            seen_models.add(mid)
            mid_context = effective_context_limit(mid)
            models_list.append(model_list_entry(mid, mid_context))
            model_registry[mid] = {
                "humanName": mid,
                "description": f"Model from {'9router' if IS_9ROUTER else 'upstream'}",
                "encoding": "o200k_base",
                "context": mid_context,
                "maxOutput": MODEL_MAX_OUTPUT_TOKENS,
            }

    all_names = ",".join(seen_models)

    return {
        "default_model": TARGET_MODEL,
        "models": models_list,
        "languages": [
            {"name": "TypeScript", "vscode_name": "typescript", "extensions": [".ts", ".tsx"]},
            {"name": "JavaScript", "vscode_name": "javascript", "extensions": [".js", ".jsx", ".mjs"]},
            {"name": "Python", "vscode_name": "python", "extensions": [".py"]},
            {"name": "Markdown", "vscode_name": "markdown", "extensions": [".md"]},
            {"name": "JSON", "vscode_name": "json", "extensions": [".json"]},
            {"name": "Go", "vscode_name": "go", "extensions": [".go"]},
            {"name": "Rust", "vscode_name": "rust", "extensions": [".rs"]},
            {"name": "Java", "vscode_name": "java", "extensions": [".java"]},
            {"name": "C++", "vscode_name": "cpp", "extensions": [".cpp", ".h", ".hpp", ".cc"]},
        ],
        "feature_flags": {
            "additional_chat_models": all_names,
            "agent_chat_model": TARGET_MODEL,
            "enable_model_registry": True,
            "model_info_registry": json.dumps(model_registry),
            "enable_hindsight": False,
            "bypass_language_filter": True,
            "small_sync_threshold": 1048576,
            "big_sync_threshold": 10485760,
            "max_upload_size_bytes": 0 if INDEXING_MODE == "complete" else 52428800,
            "cli_enable_sentry": False,
            "beachhead_enable_sentry": False,
            "use_intake_service_for_file_walk": False,
            "cli_enable_worker_thread_path_filter": False,
            "enable_prompt_enhancer": True,
            "enable_command_suggestions": True,
            "enable_subagent_support": True,
        },
        "user_tier": "ENTERPRISE_TIER",
        "user": {"id": "user_auggie_launch_local", "email": "proxy@9router.local"},
        "bootstrap_settings": {"repository_allowlist_settings": {"repository_urls": [], "is_deny_list": False}},
    }


def fake_find_missing(body: Any) -> dict[str, Any]:
    if INDEXING_MODE == "complete":
        return {"unknown_memory_names": [], "nonindexed_blob_names": []}
    if isinstance(body, dict) and isinstance(body.get("mem_object_names"), list):
        names = [name for name in body["mem_object_names"] if isinstance(name, str)]
    else:
        names = []
    return {"unknown_memory_names": names, "nonindexed_blob_names": []}


def fake_batch_upload(body: Any) -> dict[str, Any]:
    if INDEXING_MODE == "complete":
        return {"blob_names": []}
    blobs = body.get("blobs") if isinstance(body, dict) else []
    names: list[str] = []
    if isinstance(blobs, list):
        for blob in blobs:
            if isinstance(blob, dict) and isinstance(blob.get("blob_name"), str):
                names.append(blob["blob_name"])
    return {"blob_names": names}


def fake_generic(path: str) -> dict[str, Any]:
    if path == "get-credit-info":
        return {"credits": {"remaining": 999999, "used": 0, "limit": 999999}, "subscription": {"status": "active", "tier": "enterprise"}}
    if path == "get-billing-summary":
        return {"billing_summary": {"status": "active", "plan": "enterprise", "usage": 0, "limit": 999999}}
    if path == "context-canvas/list":
        return {"canvases": [], "context_canvases": [], "next_page_token": ""}
    if path.startswith("settings/"):
        return {"settings": {}, "configs": [], "permissions": [], "allowed": True}
    if path.startswith("tenant-secrets/") or path.startswith("user-secrets/"):
        return {"secrets": [], "ok": True}
    if path.startswith("remote-agents/"):
        return {"agents": [], "remote_agents": [], "messages": [], "chat_history": []}
    if path.startswith("cloud-agents/"):
        return {"agents": [], "messages": [], "ok": True}
    if path.startswith("agent-workspace/"):
        return {"updates": [], "events": [], "last_seq_id": 0, "ok": True}
    if path == "checkpoint-blobs":
        return {"new_checkpoint_id": f"checkpoint_{uuid.uuid4()}"}
    if path in {"agents/list-remote-tools", "agents/check-tool-safety"}:
        return {"tools": [], "is_safe": True, "ok": True}
    if path in {"record-user-events", "client-metrics"} or "feedback" in path:
        return {"ok": True}
    if path == "chat/exchanges/list":
        return {"chat_history": []}
    return {"ok": True}


# ============================================================================
# Local Auggie Proxy Server
# ============================================================================

def read_json(handler: BaseHTTPRequestHandler) -> Any:
    length = int(handler.headers.get("content-length") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {"_raw": raw.decode("utf-8", errors="replace")}


def extract_chat_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = text_from_value(message.get("content"))
                reasoning = text_from_value(message.get("reasoning_content") or message.get("reasoning") or message.get("thought"))
                if reasoning and STREAM_THINKING and not content.startswith("<think>"):
                    return f"<think>\n{reasoning}\n</think>\n\n{content}"
                return content
            return text_from_value(first.get("text"))
    return text_from_value(data.get("text") or data.get("content"))


def estimate_usage(request: dict[str, Any], usage: Any) -> dict[str, int]:
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or prompt + completion)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    chars = len(json.dumps(request.get("messages", []), ensure_ascii=False))
    prompt = max(1, chars // 4)
    return {"prompt_tokens": prompt, "completion_tokens": 0, "total_tokens": prompt}


def augment_usage(openai_request: dict[str, Any], usage: Any) -> dict[str, int]:
    basic = estimate_usage(openai_request, usage)
    request_model = openai_request.get("model") if isinstance(openai_request, dict) else None
    context_tokens = effective_context_limit(request_model) if isinstance(request_model, str) else MODEL_CONTEXT_TOKENS
    return {
        "input_tokens": basic["prompt_tokens"],
        "output_tokens": basic["completion_tokens"],
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "system_prompt_tokens": 0,
        "chat_history_tokens": 0,
        "current_message_tokens": basic["prompt_tokens"],
        "tool_definitions_tokens": 0,
        "tool_result_tokens": 0,
        "assistant_response_tokens": basic["completion_tokens"],
        "max_context_tokens": context_tokens or MODEL_CONTEXT_TOKENS,
        "max_output_tokens": MODEL_MAX_OUTPUT_TOKENS,
    }


def augment_chat_response(text: str, request_id: str, openai_request: dict[str, Any], usage: Any, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    token_usage = augment_usage(openai_request, usage)
    basic_usage = estimate_usage(openai_request, usage)
    nodes: list[dict[str, Any]] = []
    if text:
        nodes.append({"id": 1, "type": 0, "content": text})
    nodes.extend(tool_calls_to_nodes(tool_calls or [], starting_id=len(nodes) + 1))
    nodes.append({"id": 10000, "type": 10, "token_usage": token_usage})
    stop_reason = "tool_use" if any(node.get("type") == 5 for node in nodes) else "stop"
    return {
        "text": text,
        "response_text": text,
        "completion": text,
        "request_id": request_id,
        "requestId": request_id,
        "stop_reason": stop_reason,
        "token_usage": token_usage,
        "total_tokens": basic_usage["total_tokens"],
        "usage": basic_usage,
        "nodes": nodes,
    }


class AuggieProxy(BaseHTTPRequestHandler):
    server_version = f"auggie-launch/{__version__} (9router-native)"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if VERBOSE:
            super().log_message(fmt, *args)

    def send_json(self, value: Any, status: int = 200) -> None:
        data = json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_text(self, value: str, status: int = 200) -> None:
        data = value.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def normalized_path(self) -> str:
        parsed = urllib.parse.urlparse(self.path)
        return parsed.path.strip("/")

    def authorized(self, path: str) -> bool:
        """Rejects anything that does not carry the token injected into Auggie.

        The proxy binds to loopback only, but any local process could otherwise
        drive it (and spend upstream credits), so the token is checked on every
        request except the unauthenticated health and token endpoints.
        """
        if not REQUIRE_LOCAL_TOKEN:
            return True
        if path in {"", "health"} or path in {"token", "auth/token"} or path.endswith("/token"):
            return True
        header = self.headers.get("Authorization") or self.headers.get("authorization") or ""
        presented = header[7:].strip() if header.lower().startswith("bearer ") else header.strip()
        if not presented:
            presented = (self.headers.get("X-Api-Key") or "").strip()
        if presented and secrets.compare_digest(presented, LOCAL_TOKEN):
            return True
        log(f"rejected unauthorized local request to /{path}")
        self.send_json({"error": {"message": "unauthorized: missing or invalid local token"}}, status=401)
        return False

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.normalized_path()
        if not self.authorized(path):
            return
        if path in {"", "health"}:
            self.send_json({
                "ok": True,
                "service": "auggie-launch",
                "version": __version__,
                "router": "9router-native" if IS_9ROUTER else "standard",
                "model": TARGET_MODEL,
                "indexing_mode": INDEXING_MODE,
                "9router_combos": [c.get("name") for c in _LOCAL_9ROUTER.combos],
                "caveman": ROUTER_CAVEMAN_MODE,
            })
        elif path in {"get-models", "models", "model-config"}:
            self.send_json(fake_models())
        else:
            self.send_json(fake_generic(path))

    def do_POST(self) -> None:
        path = self.normalized_path()
        if not self.authorized(path):
            return
        body = read_json(self)
        if VERBOSE:
            log(f"{self.command} /{path}")

        if path in {"token", "auth/token"} or path.endswith("/token"):
            self.send_json(fake_token())
            return
        if path in {"get-models", "models", "model-config"}:
            self.send_json(fake_models())
            return
        if path == "find-missing":
            self.send_json(fake_find_missing(body))
            return
        if path == "batch-upload":
            self.send_json(fake_batch_upload(body))
            return
        if path in {"chat-stream", "prompt-enhancer"}:
            self.forward_stream(body)
            return
        if path in {"chat", "remote-agents/chat"}:
            self.forward_json(body)
            return
        if path in {"completion", "completion/request", "completion/complete", "chat-input-completion"}:
            self.forward_json(body)
            return
        if path in {"completion/resolve", "completion/cancel", "resolve-completions"}:
            self.send_json({"ok": True})
            return
        self.send_json(fake_generic(path))

    def forward_json(self, body: Any) -> None:
        openai_request = build_openai_request(body, stream=False)
        if VERBOSE:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            with open(os.path.join(DEBUG_DIR, "incoming_augment_request.json"), "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False, indent=2)
            with open(os.path.join(DEBUG_DIR, "outgoing_openai_request.json"), "w", encoding="utf-8") as f:
                json.dump(openai_request, f, ensure_ascii=False, indent=2)
        try:
            with open_upstream_with_retries(json_bytes(openai_request), stream=False, timeout=300, label="json") as resp:
                raw = resp.read()
            data = json.loads(raw.decode("utf-8") or "{}")
            text = extract_chat_text(data)
            tool_calls: list[dict[str, Any]] = []
            choices = data.get("choices") if isinstance(data, dict) else None
            if isinstance(choices, list) and choices:
                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
                    tool_calls = [call for call in message["tool_calls"] if isinstance(call, dict)]
            self.send_json(augment_chat_response(text, str(uuid.uuid4()), openai_request, data.get("usage") if isinstance(data, dict) else None, tool_calls))
        except urllib.error.HTTPError as exc:
            raw = compact_upstream_error(exc.msg, max_chars=2000)
            self.send_json({"error": "upstream_error", "message": raw, "status": exc.code}, status=502 if exc.code in {401, 403} else exc.code)
        except Exception as exc:
            self.send_json({"error": "upstream_error", "message": compact_upstream_error(str(exc))}, status=502)

    def forward_stream(self, body: Any) -> None:
        openai_request = build_openai_request(body, stream=True)
        if VERBOSE:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            with open(os.path.join(DEBUG_DIR, "incoming_augment_request.json"), "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False, indent=2)
            with open(os.path.join(DEBUG_DIR, "outgoing_openai_request.json"), "w", encoding="utf-8") as f:
                json.dump(openai_request, f, ensure_ascii=False, indent=2)
        request_id = str(uuid.uuid4())
        try:
            upstream_wrapper = open_upstream_with_retries(json_bytes(openai_request), stream=True, timeout=300, label="stream")
        except urllib.error.HTTPError as exc:
            msg = compact_upstream_error(exc.msg, max_chars=2000)
            self.send_json({"error": "upstream_error", "message": msg, "status": exc.code, "request_id": request_id}, status=502 if exc.code in {401, 403} else exc.code)
            return
        except Exception as exc:
            self.send_json({"error": "upstream_error", "message": compact_upstream_error(str(exc)), "request_id": request_id}, status=502)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        accumulated: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        usage: Any = None
        client_disconnected = False
        in_thinking_block = False

        def write_chunk(value: dict[str, Any]) -> bool:
            nonlocal client_disconnected
            if client_disconnected:
                return False
            payload = json_bytes(value) + b"\n"
            try:
                self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                client_disconnected = True
                log("client disconnected mid-stream; stopping upstream stream")
                return False

        try:
            with upstream_wrapper as resp:
                write_chunk({"text": "", "heartbeat": True, "request_id": request_id})
                last_heartbeat = time.time()

                for raw_line in resp:
                    if client_disconnected:
                        break

                    now = time.time()
                    if now - last_heartbeat > 5.0:
                        write_chunk({"text": "", "heartbeat": True, "request_id": request_id})
                        last_heartbeat = now

                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except Exception:
                        continue
                    if isinstance(data, dict) and data.get("usage"):
                        usage = data.get("usage")
                    choices = data.get("choices") if isinstance(data, dict) else None
                    delta_text = ""
                    reasoning_text = ""

                    if isinstance(choices, list) and choices:
                        first = choices[0]
                        if isinstance(first, dict):
                            delta = first.get("delta")
                            if isinstance(delta, dict):
                                delta_text = text_from_value(delta.get("content"))
                                reasoning_text = text_from_value(
                                    delta.get("reasoning_content")
                                    or delta.get("reasoning")
                                    or delta.get("thought")
                                )
                                raw_tool_calls = delta.get("tool_calls")
                                if isinstance(raw_tool_calls, list):
                                    tool_calls.extend(call for call in raw_tool_calls if isinstance(call, dict))
                            else:
                                delta_text = text_from_value(first.get("text"))

                    # Reasoning / thinking stream:
                    if reasoning_text:
                        if STREAM_THINKING:
                            if not in_thinking_block:
                                in_thinking_block = True
                                write_chunk({"text": "<think>\n", "delta": "<think>\n", "request_id": request_id})
                                accumulated.append("<think>\n")
                            accumulated.append(reasoning_text)
                            write_chunk({"text": reasoning_text, "delta": reasoning_text, "request_id": request_id})
                        else:
                            write_chunk({"text": "", "heartbeat": True, "thinking": True, "request_id": request_id})

                    if delta_text:
                        if in_thinking_block:
                            in_thinking_block = False
                            write_chunk({"text": "\n</think>\n\n", "delta": "\n</think>\n\n", "request_id": request_id})
                            accumulated.append("\n</think>\n\n")

                        accumulated.append(delta_text)
                        write_chunk({"text": delta_text, "delta": delta_text, "request_id": request_id})

        except Exception as exc:
            if not client_disconnected:
                write_chunk({"error": "upstream_error", "message": compact_upstream_error(str(exc)), "request_id": request_id})
        finally:
            if in_thinking_block and not client_disconnected:
                write_chunk({"text": "\n</think>\n\n", "delta": "\n</think>\n\n", "request_id": request_id})
                accumulated.append("\n</think>\n\n")

            if not client_disconnected:
                final_text = "".join(accumulated)
                final = augment_chat_response(final_text, request_id, openai_request, usage, tool_calls)
                final["done"] = True
                write_chunk(final)
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass


# ============================================================================
# Full Injections Engine (Environment, Session, MCP Tools)
# ============================================================================

def generate_injected_mcp_config() -> str:
    """Generates an MCP configuration file for Auggie CLI with 9router's tools."""
    home = os.path.expanduser("~")
    mcp_config_path = os.path.join(tempfile.gettempdir(), "auggie_injected_mcp.json")

    mcp_servers: dict[str, Any] = {}

    # Sequential thinking MCP (installed in user's global npm)
    seq_bin = os.path.join(home, ".npm-global", "bin", "mcp-server-sequential-thinking")
    if os.path.isfile(seq_bin) or os.path.islink(seq_bin):
        mcp_servers["sequential-thinking"] = {
            "command": seq_bin,
            "args": [],
        }

    # Tavily Web Search MCP if 9router has a key
    tavily_key = _LOCAL_9ROUTER.provider_api_keys.get("tavily")
    if tavily_key:
        mcp_servers["tavily"] = {
            "command": "npx",
            "args": ["-y", "@tavily/mcp-server"],
            "env": {"TAVILY_API_KEY": tavily_key},
        }

    # Exa MCP if configured
    exa_key = _LOCAL_9ROUTER.provider_api_keys.get("exa")
    if exa_key:
        mcp_servers["exa"] = {
            "command": "npx",
            "args": ["-y", "exa-mcp-server"],
            "env": {"EXA_API_KEY": exa_key},
        }

    try:
        with open(mcp_config_path, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": mcp_servers}, f, indent=2)
        return mcp_config_path
    except Exception as exc:
        log(f"could not write injected mcp config: {exc}")
        return ""


def build_injected_environment(proxy_url: str) -> dict[str, str]:
    """Builds complete environment variable injections for Auggie CLI."""
    env = os.environ.copy()
    active_key = get_active_key()

    # 1. Augment Core Injections
    env["AUGMENT_API_URL"] = proxy_url
    env["AUGMENT_API_TOKEN"] = LOCAL_TOKEN
    env["AUGMENT_SESSION_AUTH"] = json.dumps(
        {
            "accessToken": LOCAL_TOKEN,
            "tenantURL": proxy_url,
            "scopes": ["email", "profile", "offline_access"],
        },
        separators=(",", ":"),
    )
    env.setdefault("AUGMENT_INDEXING_MODE", INDEXING_MODE)
    env["AUGMENT_DISABLE_AUTO_UPDATE"] = "1"
    env["AUGMENT_MODEL"] = TARGET_MODEL
    env["AUGMENT_USER_AGENT"] = UPSTREAM_USER_AGENT

    # 2. 9router Caveman System Prompt Injections
    instructions_parts = []
    if ROUTER_CAVEMAN_MODE:
        instructions_parts.append(
            f"9router Caveman Mode Active ({ROUTER_CAVEMAN_LEVEL}): Be concise and direct. "
            "Output pure code and minimal required explanations to save tokens."
        )
    if os.environ.get("AUGMENT_INSTRUCTIONS"):
        instructions_parts.append(os.environ["AUGMENT_INSTRUCTIONS"])
    if instructions_parts:
        env["AUGMENT_INSTRUCTIONS"] = "\n\n".join(instructions_parts)

    # 3. Upstream Provider Standard Injections (for CLI tools & SDKs inside Auggie)
    env["OPENAI_BASE_URL"] = TARGET_BASE_URL
    env["OPENAI_API_KEY"] = active_key
    env["ANTHROPIC_BASE_URL"] = TARGET_BASE_URL
    env["ANTHROPIC_AUTH_TOKEN"] = active_key

    # 4. 9router Provider API Key Injections (all active providers, not just well-known)
    for prov_name, pkey in _LOCAL_9ROUTER.provider_api_keys.items():
        norm = prov_name.lower().replace("-", "_").replace(" ", "_")
        if norm == "tavily":
            env["TAVILY_API_KEY"] = pkey
        elif norm == "firecrawl":
            env["FIRECRAWL_API_KEY"] = pkey
        elif norm in ("jina_reader", "jina"):
            env["JINA_API_KEY"] = pkey
        elif norm == "minimax":
            env["MINIMAX_API_KEY"] = pkey
        elif norm in ("kilocode",):
            env["KILOCODE_TOKEN"] = pkey
        elif norm in ("gemini-cli", "gemini"):
            env["GEMINI_API_KEY"] = pkey
            env["GOOGLE_API_KEY"] = pkey
        elif norm == "qoder":
            env["QODER_TOKEN"] = pkey
        elif norm == "ollama":
            env["OLLAMA_API_KEY"] = pkey
        else:
            # Generic passthrough: PROVIDER_API_KEY
            env[f"{norm.upper()}_API_KEY"] = pkey

    # 5. Ensure global npm bin and local bin are on PATH
    home = os.path.expanduser("~")
    extra_paths = [
        os.path.join(home, ".npm-global", "bin"),
        os.path.join(home, ".local", "bin"),
        "/usr/local/bin",
    ]
    cur_path = env.get("PATH", "")
    for ep in extra_paths:
        if os.path.isdir(ep) and ep not in cur_path.split(":"):
            cur_path = f"{ep}:{cur_path}"
    env["PATH"] = cur_path

    return env


# ============================================================================
# 9router Helper & CLI Diagnostics Commands
# ============================================================================

def print_9router_combos() -> None:
    """Lists all combos and their constituent models from 9router."""
    print("=" * 60)
    print("🔀 9ROUTER COMBOS & MODEL ALIASES")
    print("=" * 60)
    if _LOCAL_9ROUTER.combos:
        print(f"Active Combos ({len(_LOCAL_9ROUTER.combos)}):")
        for c in _LOCAL_9ROUTER.combos:
            cname = c.get("name")
            models = c.get("models") or []
            print(f"  • {cname} ({len(models)} fallback models):")
            for m in models:
                print(f"      - {m}")
    else:
        print("  (No combos defined in ~/.9router/db.json)")

    if _LOCAL_9ROUTER.model_aliases:
        print(f"\nModel Aliases ({len(_LOCAL_9ROUTER.model_aliases)}):")
        for alias, real in _LOCAL_9ROUTER.model_aliases.items():
            print(f"  • {alias:25} -> {real}")
    print("=" * 60)


def print_9router_stats() -> None:
    """Queries 9router's /api/usage and prints token savings and stats."""
    print("=" * 60)
    print("📊 9ROUTER USAGE & SAVINGS STATISTICS")
    print("=" * 60)

    # 1. Local state overview
    print(f"9router Installation : {'Found (~/.9router)' if _LOCAL_9ROUTER.installed else 'Not detected'}")
    print(f"Caveman Mode         : {'Enabled' if _LOCAL_9ROUTER.caveman_enabled else 'Disabled'} (Level: {_LOCAL_9ROUTER.caveman_level})")
    print(f"Tunnel URL           : {_LOCAL_9ROUTER.tunnel_url or '(none)'}")
    print(f"Configured Providers : {len(_LOCAL_9ROUTER.provider_connections)} ({', '.join(_LOCAL_9ROUTER.provider_api_keys.keys())})")

    # 2. Try querying live /api/usage
    parsed = urllib.parse.urlparse(TARGET_BASE_URL)
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
_BUNDLED_DB_BACKUP_DIR = os.path.join(_HERE, "9router", "db")


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
    if binary and _LOCAL_9ROUTER.installed:
        return True

    if not binary:
        print("9router not detected on this system.")
        if not install_9router():
            return False

    restore_9router_db()
    return find_9router_binary() is not None


def _ping_tunnel(tunnel_url: str) -> bool:
    """Ping tunnel URL to check if reachable. Returns True if reachable."""
    try:
        t_parsed = urllib.parse.urlparse(tunnel_url)
        t_host = t_parsed.hostname or "localhost"
        t_port = t_parsed.port or (443 if t_parsed.scheme == "https" else 80)
        ts = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ts.settimeout(3.0)
        ts.connect((t_host, t_port))
        ts.close()
        return True
    except Exception:
        return False


def run_doctor_check() -> int:
    """Performs deep health and connectivity verification for 9router and Auggie."""
    print("=" * 60)
    print("🔍 AUGGIE-LAUNCH HEALTH & 9ROUTER DIAGNOSTICS")
    print("=" * 60)

    # 1. Config summary
    print(f"Target Base URL : {TARGET_BASE_URL}")
    print(f"Target Model    : {TARGET_MODEL}")
    print(f"API Key         : {API_KEYS[0][:10]}... ({len(API_KEYS)} key(s) loaded)")
    print(f"9router Detected: {'YES (native local + host match)' if IS_9ROUTER else 'NO'}")
    print(f"Local 9router DB: {'Found (~/.9router/db.json)' if _LOCAL_9ROUTER.installed else 'Not found'}")
    if _LOCAL_9ROUTER.tunnel_url:
        print(f"Cloudflare Tunl : {_LOCAL_9ROUTER.tunnel_url}")
    print(f"Caveman Mode    : {'Enabled (' + ROUTER_CAVEMAN_LEVEL + ')' if ROUTER_CAVEMAN_MODE else 'Disabled'}")
    print(f"Reasoning Stream: {STREAM_THINKING}")
    print("-" * 60)

    # 2. Upstream Network & Ping Test
    print("1. Testing connection to Upstream / 9router...")
    parsed = urllib.parse.urlparse(TARGET_BASE_URL)
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
        if _LOCAL_9ROUTER.tunnel_url:
            if _ping_tunnel(_LOCAL_9ROUTER.tunnel_url):
                t_parsed = urllib.parse.urlparse(_LOCAL_9ROUTER.tunnel_url)
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
            print(f"   [INFO] Discovered {len(_LOCAL_9ROUTER.combos)} combos & {len(_LOCAL_9ROUTER.model_aliases)} aliases in local 9router DB.")
    except Exception as exc:
        print(f"   [WARN] Could not query /models: {exc}")

    # 4. Minimal Completion Test
    if connected:
        print(f"3. Testing completion ping with model '{TARGET_MODEL}'...")
        req_body = {
            "model": TARGET_MODEL,
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
        res = subprocess.run([AUGGIE_BIN, "--version"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            print(f"   [OK] Auggie binary found: {AUGGIE_BIN} -> {res.stdout.strip()}")
        else:
            print(f"   [WARN] Auggie returned exit code {res.returncode}: {res.stderr.strip()}")
    except FileNotFoundError:
        print(f"   [FAIL] Auggie CLI '{AUGGIE_BIN}' not found on PATH.")
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
    if _LOCAL_9ROUTER.combos:
        print("  [9router Combos]:")
        for c in _LOCAL_9ROUTER.combos:
            cname = c.get("name")
            seen.add(cname)
            print(f"    - {cname}")
    if _LOCAL_9ROUTER.model_aliases:
        print("  [9router Aliases]:")
        for a, r in _LOCAL_9ROUTER.model_aliases.items():
            seen.add(a)
            print(f"    - {a} -> {r}")
    if models:
        print("  [Live Registry Models]:")
        for m in models:
            mid = m.get("id") or m.get("name")
            if mid not in seen:
                print(f"    - {mid}")


def print_env(port: int) -> None:
    print(f"AUGGIE_BIN={AUGGIE_BIN}")
    print(f"AUGGIE_LAUNCH_PROXY_URL=http://127.0.0.1:{port}")
    print(f"AUGGIE_LAUNCH_BASE_URL={TARGET_BASE_URL}")
    print(f"AUGGIE_LAUNCH_MODEL={TARGET_MODEL}")
    print(f"AUGGIE_LAUNCH_API_KEY={'<set>' if API_KEYS else ''}")
    print(f"AUGGIE_LAUNCH_IS_9ROUTER={'true' if IS_9ROUTER else 'false'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_LOCAL_DB={'true' if _LOCAL_9ROUTER.installed else 'false'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_CAVEMAN={'true (' + ROUTER_CAVEMAN_LEVEL + ')' if ROUTER_CAVEMAN_MODE else 'false'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_TUNNEL={_LOCAL_9ROUTER.tunnel_url or '(none)'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_COMBOS={','.join(str(c.get('name')) for c in _LOCAL_9ROUTER.combos if c.get('name')) or '(none)'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_ALIASES={','.join(_LOCAL_9ROUTER.model_aliases.keys()) or '(none)'}")
    print(f"AUGGIE_LAUNCH_9ROUTER_PROVIDERS={','.join(_LOCAL_9ROUTER.provider_api_keys.keys()) or '(none)'}")
    print(f"AUGGIE_LAUNCH_INDEXING_MODE={INDEXING_MODE}")
    print(f"AUGGIE_LAUNCH_STREAM_THINKING={'true' if STREAM_THINKING else 'false'}")
    print(f"AUGGIE_LAUNCH_USER_AGENT={UPSTREAM_USER_AGENT}")
    print(f"AUGGIE_LAUNCH_UPSTREAM_APP_NAME={UPSTREAM_APP_NAME}")
    print(f"AUGGIE_LAUNCH_MODEL_CONTEXT={model_context_limit(TARGET_MODEL)}")
    print("loaded_env_files=" + (", ".join(_LOADED_ENV_FILES) if _LOADED_ENV_FILES else "(none)"))


def stop_server(httpd: ThreadingHTTPServer) -> None:
    """Shuts the proxy down without ever blocking the exit path (e.g. on Ctrl+C)."""
    stopper = threading.Thread(target=httpd.shutdown, daemon=True)
    stopper.start()
    stopper.join(timeout=2.0)
    try:
        httpd.server_close()
    except Exception:
        pass


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
    port = PORT or find_free_port()

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

    if VERBOSE:
        log(f"proxy={proxy_url} upstream={upstream_url()} model={TARGET_MODEL} 9router={IS_9ROUTER} indexing={INDEXING_MODE}")
        if _LOADED_ENV_FILES:
            log("env files: " + ", ".join(_LOADED_ENV_FILES))

    if "--proxy-only" in launcher_args:
        try:
            print(f"Proxy ready at {proxy_url} (Forwarding to: {TARGET_BASE_URL})")
            thread.join()
        except KeyboardInterrupt:
            pass
        finally:
            stop_server(httpd)
        return

    # Build full injections into Auggie CLI environment
    env = build_injected_environment(proxy_url)

    # Injected MCP config if enabled and not already provided
    if AUTO_INJECT_MCP and "--mcp-config" not in pass_args:
        mcp_path = generate_injected_mcp_config()
        if mcp_path and os.path.isfile(mcp_path):
            pass_args = ["--mcp-config", mcp_path, *pass_args]
            log(f"injected MCP tools config from 9router: {mcp_path}")

    try:
        result = subprocess.run([AUGGIE_BIN, *pass_args], env=env)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"error: cannot find auggie binary ({AUGGIE_BIN})", file=sys.stderr)
        print("Tip: Ensure auggie is installed or run ./install.sh", file=sys.stderr)
        sys.exit(127)
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        stop_server(httpd)


if __name__ == "__main__":
    main()
