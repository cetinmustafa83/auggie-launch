from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from . import config
from .config import log
from .transform import text_from_value

# ============================================================================
# CodeGPT Plus Cloud Bridge
# ============================================================================
#
# CodeGPT Plus exposes an OpenAI-shaped SSE stream, but it routes every chat
# through an *agent* and authenticates with the short-lived session token the
# VS Code extension serves on http://localhost:54112/api/session.
#
# Two things differ from a plain OpenAI upstream:
#   1. The chat endpoint is /chat/extension (or /chat/tools for function calls).
#   2. The body carries `agentId` (plus statusId/requestId), not just `model`.
#
# This module keeps the token fresh and hands the proxy the headers/body shims
# it needs, so the rest of the pipeline keeps speaking plain OpenAI.

_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE: dict[str, Any] = {"token": "", "fetched_at": 0.0, "session": {}}
_TOKEN_TTL = 60.0


def _session_payload() -> dict[str, Any]:
    """Fetches the live session JSON from the local CodeGPT sidecar."""
    url = config.CODEGPT_SESSION_URL
    if not url:
        return {}
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
            return data if isinstance(data, dict) else {}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log(f"codegpt: could not read session from {url}: {exc}")
        return {}


def session_token() -> str:
    """Returns a CodeGPT access token, pinned statically or fetched per session.

    When AUGGIE_LAUNCH_CODEGPT_TOKEN is set the token is used as-is and the local
    sidecar is never contacted, so the proxy runs without VS Code open.
    """
    if not config.IS_CODEGPT:
        return ""
    if config.CODEGPT_TOKEN:
        return config.CODEGPT_TOKEN
    now = time.time()
    with _TOKEN_LOCK:
        cached = str(_TOKEN_CACHE.get("token") or "")
        age = now - float(_TOKEN_CACHE.get("fetched_at") or 0.0)
        if cached and age < _TOKEN_TTL:
            return cached
    session = _session_payload()
    token = str(session.get("accessToken") or "").strip()
    with _TOKEN_LOCK:
        if token:
            _TOKEN_CACHE["token"] = token
            _TOKEN_CACHE["fetched_at"] = now
            _TOKEN_CACHE["session"] = session
        elif cached:
            # Keep the last good token rather than dropping to an empty one.
            return cached
        return str(_TOKEN_CACHE.get("token") or "")


def session_field(name: str, env_value: str) -> str:
    """Prefers an explicit config/env value, falling back to the live session."""
    if env_value:
        return env_value
    if config.CODEGPT_TOKEN:
        # Pinned token: the sidecar is not needed, so do not probe it.
        return ""
    session = _TOKEN_CACHE.get("session")
    if not isinstance(session, dict) or not session:
        session = _session_payload()
        if session:
            with _TOKEN_LOCK:
                _TOKEN_CACHE["session"] = session
    value = session.get(name) if isinstance(session, dict) else None
    return str(value or "").strip()


def chat_path(has_tools: bool) -> str:
    """CodeGPT routes tool calls and plain chats through different agents paths."""
    return "/chat/tools" if has_tools else "/chat/extension"


def extra_headers() -> dict[str, str]:
    """Builds the CodeGPT-specific header set (auth + identity + routing hints)."""
    headers: dict[str, str] = {
        "tokens": "true",
        "source": "api",
        "channel": "api",
        "codegpt-version": config.CODEGPT_VERSION,
    }
    token = session_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    distinct = session_field("distinctId", config.CODEGPT_DISTINCT_ID)
    if distinct:
        headers["distinct-id"] = distinct
    signed = session_field("signedDistinctId", config.CODEGPT_SIGNED_DISTINCT_ID)
    if signed:
        headers["X-Signed-Distinct-Id"] = signed
    org = config.CODEGPT_ORG_ID
    if org:
        headers["CodeGPT-Org-Id"] = org
    return headers


def gemini_safe_schema(value: Any) -> Any:
    """Makes a JSON Schema acceptable to CodeGPT's Vertex/Gemini backend.

    Gemini rejects non-string `enum` members (e.g. [1, 2, 3]) with
    INVALID_ARGUMENT, while Auggie's MCP tools do emit integer enums. Coercing
    the values (and the accompanying type) to string keeps the tool callable.
    """
    if isinstance(value, list):
        return [gemini_safe_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, val in value.items():
        if key == "enum" and isinstance(val, list) and any(not isinstance(item, str) for item in val):
            out[key] = [str(item) for item in val]
            out["type"] = "string"
            continue
        out[key] = gemini_safe_schema(val)
    return out


def _flatten_tool_calls(tool_calls: Any) -> str:
    """Renders assistant tool_calls as plain text for the Gemini backend."""
    lines: list[str] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = fn.get("name") or call.get("name") or "tool"
            args = fn.get("arguments") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments") or {})
            lines.append(f"[tool call {name}({args})]")
    return "\n".join(lines)


def gemini_safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repairs a message list that Vertex/Gemini would reject with 400.

    Three Vertex quirks are handled here:
      * a conversation ending on a model turn is rejected ("Requests ending with
        a model turn are not supported"), so a user nudge is appended;
      * `role: "tool"` messages and `assistant.tool_calls` are not matched up the
        way OpenAI does, causing "number of function response parts is not equal
        to the number of function call parts" -- tool traffic is folded into
        plain user text instead;
      * an assistant turn carrying only tool_calls has null content, which reads
        as an empty turn.
    """
    if not messages:
        return [{"role": "user", "content": "(empty request)"}]

    out: list[dict[str, Any]] = []
    pending: list[str] = []

    def flush_pending() -> None:
        if pending:
            out.append({"role": "user", "content": "\n".join(pending)})
            pending.clear()

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        if role == "tool":
            content = msg.get("content")
            if isinstance(content, list):
                content = text_from_value(content)
            pending.append(f"[tool result] {content if content is not None else ''}")
            continue
        if role in {"assistant", "model"}:
            calls = _flatten_tool_calls(msg.get("tool_calls"))
            text = msg.get("content")
            if isinstance(text, list):
                text = text_from_value(text)
            combined = "\n".join(part for part in (text or "", calls) if part).strip()
            if combined:
                flush_pending()
                out.append({"role": "assistant", "content": combined})
            elif not calls:
                flush_pending()
                out.append({"role": "assistant", "content": text or ""})
            continue
        flush_pending()
        text = msg.get("content")
        if isinstance(text, list):
            text = text_from_value(text)
        out.append({"role": role or "user", "content": text if text is not None else ""})
    flush_pending()

    if not out:
        return [{"role": "user", "content": "(empty request)"}]
    while out and str(out[-1].get("role") or "").lower() in {"assistant", "model"}:
        out.append({"role": "user", "content": "Continue."})
    return out


def adapt_request_body(request: dict[str, Any]) -> dict[str, Any]:
    """Rewrites an OpenAI chat body into the CodeGPT agent request shape.

    OpenAI tools use {type, function:{name, description, parameters}}; CodeGPT
    expects a flat {name, description, parameters}. `agentId` replaces `model`.
    """
    body: dict[str, Any] = {
        "messages": gemini_safe_messages(request.get("messages") or []),
        "temperature": request.get("temperature", 0),
        "stream": True,
        "format": "json",
        "requestId": str(uuid.uuid4()),
    }
    agent_id = config.CODEGPT_AGENT_ID
    if agent_id:
        body["agentId"] = agent_id

    tools = request.get("tools")
    if isinstance(tools, list) and tools:
        flat_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            flat_tools.append({
                "name": name,
                "description": str(fn.get("description") or ""),
                "parameters": gemini_safe_schema(fn.get("parameters") or {"type": "object", "properties": {}}),
            })
        if flat_tools:
            body["tools"] = flat_tools
    return body
