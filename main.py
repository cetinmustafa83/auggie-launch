#!/usr/bin/env python3
# Copyright (c) 2026 auggie-launch contributors
# For learning and research only; any other use is at your own risk.
"""auggie-launch: local Python proxy and launcher for Auggie/Augment Code CLI.

Starts a local Augment-compatible proxy, disables Auggie RAG/indexing by default,
forwards chat/completion requests to an OpenAI-compatible Chat Completions API,
and launches the installed `auggie` binary with local Augment session env vars.
"""

from __future__ import annotations

from email.utils import parsedate_to_datetime
import json
import os
import random
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))

_REQUIRED_KEYS = (
    "AUGGIE_LAUNCH_BASE_URL",
    "AUGGIE_LAUNCH_MODEL",
)

TARGET_BASE_URL = ""
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
MODEL_MAX_OUTPUT_TOKENS = 16000
REASONING_EFFORT = ""
_LOADED_ENV_FILES: list[str] = []


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


def load_dotenv_files() -> list[str]:
    loaded: list[str] = []
    claimed = set(os.environ.keys())
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


def env_int(name: str, fallback: int) -> int:
    try:
        value = int((os.environ.get(name) or "").strip())
        return value if value > 0 else fallback
    except ValueError:
        return fallback


def load_config() -> None:
    global TARGET_BASE_URL, TARGET_MODEL, TARGET_API_KEY, API_KEYS, AUGGIE_BIN
    global LOCAL_TOKEN, VERBOSE, DEBUG_DIR, PORT, _LOADED_ENV_FILES
    global UPSTREAM_USER_AGENT, UPSTREAM_APP_NAME, SANITIZE_UPSTREAM_PROMPTS
    global UPSTREAM_MIN_INTERVAL_SECONDS, UPSTREAM_RETRIES
    global UPSTREAM_429_FREEZE_SECONDS, UPSTREAM_5XX_FREEZE_SECONDS
    global UPSTREAM_MAX_RETRY_AFTER_SECONDS, UPSTREAM_BACKOFF_INITIAL_SECONDS
    global UPSTREAM_BACKOFF_MAX_SECONDS
    global INDEXING_MODE, MODEL_CONTEXT_TOKENS, MODEL_MAX_OUTPUT_TOKENS, REASONING_EFFORT

    _LOADED_ENV_FILES = load_dotenv_files()

    missing = [key for key in _REQUIRED_KEYS if not (os.environ.get(key) or "").strip()]
    key_text = os.environ.get("AUGGIE_LAUNCH_API_KEYS") or os.environ.get("AUGGIE_LAUNCH_API_KEY") or ""
    API_KEYS = [key.strip() for key in key_text.split(",") if key.strip()]
    if not API_KEYS:
        missing.append("AUGGIE_LAUNCH_API_KEY (or AUGGIE_LAUNCH_API_KEYS)")
    if missing:
        xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        print("error: missing required configuration: " + ", ".join(missing), file=sys.stderr)
        print("Set them in ./.env or ~/.config/auggie-launch/.env", file=sys.stderr)
        print(f"template: {os.path.join(_HERE, '.env.example')}", file=sys.stderr)
        if _LOADED_ENV_FILES:
            print("loaded env files:", file=sys.stderr)
            for path in _LOADED_ENV_FILES:
                print(f"  - {path}", file=sys.stderr)
        sys.exit(2)

    TARGET_BASE_URL = os.environ["AUGGIE_LAUNCH_BASE_URL"].strip().rstrip("/")
    TARGET_MODEL = os.environ["AUGGIE_LAUNCH_MODEL"].strip()
    TARGET_API_KEY = API_KEYS[0]
    AUGGIE_BIN = (os.environ.get("AUGGIE_BIN") or "auggie").strip()
    LOCAL_TOKEN = (os.environ.get("AUGGIE_LAUNCH_LOCAL_TOKEN") or "fake-augment-access-token").strip()
    VERBOSE = env_truthy("AUGGIE_LAUNCH_VERBOSE")
    DEBUG_DIR = (os.environ.get("AUGGIE_LAUNCH_DEBUG_DIR") or os.path.join(tempfile.gettempdir(), "auggie-launch")).strip()
    PORT = env_int("AUGGIE_LAUNCH_PORT", 0)
    UPSTREAM_USER_AGENT = (os.environ.get("AUGGIE_LAUNCH_USER_AGENT") or "codex-cli").strip()
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
    MODEL_MAX_OUTPUT_TOKENS = env_int("AUGGIE_LAUNCH_MODEL_MAX_OUTPUT_TOKENS", 16000)
    REASONING_EFFORT = (os.environ.get("AUGGIE_LAUNCH_REASONING_EFFORT") or "").strip().lower()
    if REASONING_EFFORT and REASONING_EFFORT not in {"low", "medium", "high"}:
        print("error: AUGGIE_LAUNCH_REASONING_EFFORT must be low, medium, or high", file=sys.stderr)
        sys.exit(2)


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
    base = min(UPSTREAM_BACKOFF_MAX_SECONDS, UPSTREAM_BACKOFF_INITIAL_SECONDS * (2 ** max(0, attempt)))
    return base + random.uniform(0.0, min(1.0, base * 0.25))


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
        return API_KEYS[0]


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


def log(message: str) -> None:
    if VERBOSE:
        print(f"[auggie-launch] {message}", file=sys.stderr)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


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


def collect_strings(value: Any, *, limit: int = 48) -> list[str]:
    found: list[str] = []

    def walk(node: Any) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, str):
            stripped = node.strip()
            if stripped:
                found.append(stripped)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, dict):
            preferred = ("message", "prompt", "text", "content", "query", "input", "instruction")
            for key in preferred:
                if key in node:
                    walk(node[key])
            for key, item in node.items():
                if key not in preferred:
                    walk(item)

    walk(value)
    return found


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


def build_openai_messages(body: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = os.environ.get("AUGGIE_LAUNCH_SYSTEM_PROMPT", "").strip()
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


def merge_stream_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    last_key = ""
    anon = 0
    for call in tool_calls:
        index = call.get("index")
        call_id = call.get("id") if isinstance(call.get("id"), str) and call.get("id") else None
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn.get("name"), str) and fn.get("name") else None
        args = fn.get("arguments") if isinstance(fn.get("arguments"), str) else None
        if call_id and call_id in merged_by_key:
            key = call_id
        elif isinstance(index, int):
            key = f"index:{index}"
        elif call_id:
            key = call_id
        elif last_key and not name:
            key = last_key
        else:
            key = f"anon:{anon}"
            anon += 1
        if key not in merged_by_key:
            merged_by_key[key] = {"id": call_id or f"tool_{uuid.uuid4().hex[:12]}", "type": "function", "function": {}}
            order.append(key)
        merged = merged_by_key[key]
        if call_id:
            merged["id"] = call_id
        merged_fn = merged.setdefault("function", {})
        if name:
            old_name = merged_fn.get("name") if isinstance(merged_fn.get("name"), str) else ""
            merged_fn["name"] = name if not old_name or name.startswith(old_name) else old_name
        if args is not None:
            old_args = merged_fn.get("arguments") if isinstance(merged_fn.get("arguments"), str) else ""
            merged_fn["arguments"] = args if not old_args or args.startswith(old_args) else old_args + args
        last_key = key
    return [merged_by_key[key] for key in order if isinstance(merged_by_key[key].get("function"), dict) and merged_by_key[key]["function"].get("name")]


def tool_calls_to_nodes(tool_calls: list[dict[str, Any]], *, starting_id: int = 2) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    node_id = starting_id
    for call in merge_stream_tool_calls(tool_calls):
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name")
        args = fn.get("arguments") if isinstance(fn.get("arguments"), str) else "{}"
        call_id = call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        if not isinstance(name, str) or not name:
            continue
        nodes.append({"id": node_id, "type": 5, "tool_use": {"tool_name": name, "tool_use_id": str(call_id), "input_json": args}})
        node_id += 1
    return nodes


def build_openai_request(body: Any, *, stream: bool) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": TARGET_MODEL,
        "messages": build_openai_messages(body),
        "stream": stream,
    }
    tools = build_openai_tools(body)
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"
    if stream:
        request["stream_options"] = {"include_usage": True}
    if isinstance(body, dict):
        if isinstance(body.get("temperature"), (int, float)):
            request["temperature"] = body["temperature"]
        max_tokens = body.get("max_tokens") or body.get("max_output_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            request["max_tokens"] = max_tokens
        if isinstance(body.get("reasoning_effort"), str) and body["reasoning_effort"].strip():
            request["reasoning_effort"] = body["reasoning_effort"].strip().lower()
    if "reasoning_effort" not in request and REASONING_EFFORT:
        request["reasoning_effort"] = REASONING_EFFORT
    return request


def upstream_headers(api_key: str, *, stream: bool) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if UPSTREAM_USER_AGENT:
        headers["User-Agent"] = UPSTREAM_USER_AGENT
    return headers


def upstream_url() -> str:
    return f"{TARGET_BASE_URL}/chat/completions"


def upstream_request(body: Any, *, stream: bool) -> urllib.request.Request:
    request_body = build_openai_request(body, stream=stream)
    if VERBOSE:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        with open(os.path.join(DEBUG_DIR, "outgoing_openai_request.json"), "w", encoding="utf-8") as f:
            json.dump(request_body, f, ensure_ascii=False, indent=2)
    api_key = get_active_key()
    data = json_bytes(request_body)
    req = urllib.request.Request(upstream_url(), data=data, method="POST", headers=upstream_headers(api_key, stream=stream))
    req.add_header("X-Auggie-Launch-Key", api_key)
    return req


def open_upstream_with_retries(data: bytes, *, stream: bool, timeout: int, label: str) -> Any:
    last_error: Exception | None = None
    max_attempts = max(len(API_KEYS), 1) + max(0, UPSTREAM_RETRIES)
    for attempt in range(max_attempts):
        api_key = get_active_key()
        log(f"{label} upstream attempt={attempt + 1}/{max_attempts} key={api_key[:10]}...")
        req = urllib.request.Request(
            upstream_url(),
            data=data,
            method="POST",
            headers=upstream_headers(api_key, stream=stream),
        )
        try:
            wait_for_upstream_slot()
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            frozen_for = mark_key_failed(api_key, exc.code, retry_after)
            last_error = exc
            if exc.code == 429:
                cooldown = clamp_retry_delay(
                    parse_retry_after(retry_after),
                    min(frozen_for or UPSTREAM_429_FREEZE_SECONDS, retry_backoff_seconds(attempt)),
                )
                apply_upstream_cooldown(cooldown, "429 rate limit")
            elif 500 <= exc.code <= 599:
                apply_upstream_cooldown(retry_backoff_seconds(attempt), f"{exc.code} upstream error")
            if is_retryable_upstream_status(exc.code) and attempt < max_attempts - 1:
                continue
            if exc.code in (401, 402) and attempt < min(len(API_KEYS), max_attempts) - 1:
                continue
            break
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                apply_upstream_cooldown(retry_backoff_seconds(attempt), "transport error")
                continue
            break
    if last_error is not None:
        raise last_error
    raise RuntimeError("upstream request failed")


def extract_chat_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                return text_from_value(message.get("content"))
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
        "max_context_tokens": MODEL_CONTEXT_TOKENS,
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


def fake_token() -> dict[str, Any]:
    return {
        "access_token": LOCAL_TOKEN,
        "token_type": "Bearer",
        "expires_in": 31536000,
        "scope": "email profile offline_access",
    }


def fake_models() -> dict[str, Any]:
    return {
        "default_model": TARGET_MODEL,
        "models": [
            {
                "name": TARGET_MODEL,
                "internal_name": TARGET_MODEL,
                "suggested_prefix_char_count": 12000,
                "suggested_suffix_char_count": 12000,
                "completion_timeout_ms": 600000,
            }
        ],
        "languages": [
            {"name": "TypeScript", "vscode_name": "typescript", "extensions": [".ts", ".tsx"]},
            {"name": "JavaScript", "vscode_name": "javascript", "extensions": [".js", ".jsx", ".mjs"]},
            {"name": "Python", "vscode_name": "python", "extensions": [".py"]},
            {"name": "Markdown", "vscode_name": "markdown", "extensions": [".md"]},
            {"name": "JSON", "vscode_name": "json", "extensions": [".json"]},
        ],
        "feature_flags": {
            "additional_chat_models": TARGET_MODEL,
            "agent_chat_model": TARGET_MODEL,
            "enable_model_registry": True,
            "model_info_registry": json.dumps({
                TARGET_MODEL: {
                    "humanName": TARGET_MODEL,
                    "description": "OpenAI-compatible upstream model via auggie-launch",
                    "encoding": "o200k_base",
                    "context": MODEL_CONTEXT_TOKENS,
                    "maxOutput": MODEL_MAX_OUTPUT_TOKENS,
                }
            }),
            "enable_hindsight": False,
            "bypass_language_filter": True,
            "small_sync_threshold": 1048576,
            "big_sync_threshold": 10485760,
            "max_upload_size_bytes": 0 if INDEXING_MODE == "complete" else 52428800,
            "cli_enable_sentry": False,
            "beachhead_enable_sentry": False,
            "use_intake_service_for_file_walk": False,
            "cli_enable_worker_thread_path_filter": False,
        },
        "user_tier": "ENTERPRISE_TIER",
        "user": {"id": "user_auggie_launch_local", "email": "proxy@example.local"},
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
    # Default complete mode forbids RAG/blob ingestion: acknowledge without
    # storing or indexing content. This prevents workspace RAG upload by default.
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


class AuggieProxy(BaseHTTPRequestHandler):
    server_version = "auggie-launch/0.1"
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
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, value: str, status: int = 200) -> None:
        data = value.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def normalized_path(self) -> str:
        parsed = urllib.parse.urlparse(self.path)
        return parsed.path.strip("/")

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.normalized_path()
        if path in {"", "health"}:
            self.send_json({"ok": True, "service": "auggie-launch", "indexing_mode": INDEXING_MODE})
        elif path in {"get-models", "models", "model-config"}:
            self.send_json(fake_models())
        else:
            self.send_json(fake_generic(path))

    def do_POST(self) -> None:
        path = self.normalized_path()
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
            raw = compact_upstream_error(exc.read().decode("utf-8", errors="replace"), max_chars=2000)
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
            upstream_resp = open_upstream_with_retries(json_bytes(openai_request), stream=True, timeout=300, label="stream")
        except urllib.error.HTTPError as exc:
            msg = compact_upstream_error(exc.read().decode("utf-8", errors="replace"), max_chars=2000)
            self.send_json({"error": "upstream_error", "message": msg, "status": exc.code, "request_id": request_id}, status=502 if exc.code in {401, 403} else exc.code)
            return
        except Exception as exc:
            self.send_json({"error": "upstream_error", "message": compact_upstream_error(str(exc)), "request_id": request_id}, status=502)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        accumulated: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        usage: Any = None

        def write_chunk(value: dict[str, Any]) -> None:
            payload = json_bytes(value) + b"\n"
            self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        try:
            with upstream_resp as resp:
                write_chunk({"text": "", "heartbeat": True, "request_id": request_id})
                for raw_line in resp:
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
                    if isinstance(choices, list) and choices:
                        first = choices[0]
                        if isinstance(first, dict):
                            delta = first.get("delta")
                            if isinstance(delta, dict):
                                delta_text = text_from_value(delta.get("content"))
                                raw_tool_calls = delta.get("tool_calls")
                                if isinstance(raw_tool_calls, list):
                                    tool_calls.extend(call for call in raw_tool_calls if isinstance(call, dict))
                            else:
                                delta_text = text_from_value(first.get("text"))
                    if delta_text:
                        accumulated.append(delta_text)
                        write_chunk({"text": delta_text, "delta": delta_text, "request_id": request_id})
        except Exception as exc:
            write_chunk({"error": "upstream_error", "message": compact_upstream_error(str(exc)), "request_id": request_id})
        finally:
            final_text = "".join(accumulated)
            final = augment_chat_response(final_text, request_id, openai_request, usage, tool_calls)
            final["done"] = True
            write_chunk(final)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()


def print_env(port: int) -> None:
    print(f"AUGGIE_BIN={AUGGIE_BIN}")
    print(f"AUGGIE_LAUNCH_PROXY_URL=http://127.0.0.1:{port}")
    print(f"AUGGIE_LAUNCH_BASE_URL={TARGET_BASE_URL}")
    print(f"AUGGIE_LAUNCH_MODEL={TARGET_MODEL}")
    print(f"AUGGIE_LAUNCH_API_KEY={'<set>' if API_KEYS else ''}")
    print(f"AUGGIE_LAUNCH_INDEXING_MODE={INDEXING_MODE}")
    print(f"AUGGIE_LAUNCH_USER_AGENT={UPSTREAM_USER_AGENT}")
    print(f"AUGGIE_LAUNCH_UPSTREAM_APP_NAME={UPSTREAM_APP_NAME}")
    print(f"AUGGIE_LAUNCH_SANITIZE_UPSTREAM_PROMPTS={str(SANITIZE_UPSTREAM_PROMPTS).lower()}")
    print("loaded_env_files=" + (", ".join(_LOADED_ENV_FILES) if _LOADED_ENV_FILES else "(none)"))


def main() -> None:
    launcher_args = []
    pass_args = []
    args = sys.argv[1:]
    while args:
        arg = args.pop(0)
        if arg == "--":
            pass_args.extend(args)
            break
        if arg in {"--print-env", "--proxy-only", "--help", "-h"}:
            launcher_args.append(arg)
        else:
            pass_args.append(arg)

    if "--help" in launcher_args or "-h" in launcher_args:
        print("Usage: auggie-launch [--print-env] [--proxy-only] -- [auggie args]")
        print("       auggie-launch [auggie args]")
        return

    load_config()
    port = PORT or find_free_port()

    if "--print-env" in launcher_args:
        print_env(port)
        return

    httpd = ThreadingHTTPServer(("127.0.0.1", port), AuggieProxy)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    proxy_url = f"http://127.0.0.1:{port}"

    if VERBOSE:
        log(f"proxy={proxy_url} upstream={upstream_url()} model={TARGET_MODEL} indexing={INDEXING_MODE}")
        if _LOADED_ENV_FILES:
            log("env files: " + ", ".join(_LOADED_ENV_FILES))

    if "--proxy-only" in launcher_args:
        try:
            print(proxy_url)
            thread.join()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.shutdown()
            httpd.server_close()
        return

    env = os.environ.copy()
    env["AUGMENT_API_URL"] = proxy_url
    env["AUGMENT_API_TOKEN"] = LOCAL_TOKEN
    env["AUGMENT_SESSION_AUTH"] = json.dumps(
        {"accessToken": LOCAL_TOKEN, "tenantURL": proxy_url, "scopes": ["email", "profile", "offline_access"]},
        separators=(",", ":"),
    )
    env.setdefault("AUGMENT_INDEXING_MODE", INDEXING_MODE)

    try:
        result = subprocess.run([AUGGIE_BIN, *pass_args], env=env)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"error: cannot find auggie binary ({AUGGIE_BIN})", file=sys.stderr)
        sys.exit(127)
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
