from __future__ import annotations

import json
import re
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


def chat_path(has_tools: bool = True) -> str:
    """The CodeGPT Plus free-tier bridge path.

    Discovered from the extension's own traffic: the inclusive-model endpoint is
    `/chat/tools/<harness>`, where `<harness>` is the client name. It takes a
    `modelId` plus an `X-Provider` header and needs no agent at all.
    """
    return f"/chat/tools/{config.CODEGPT_HARNESS}"


def extra_headers() -> dict[str, str]:
    """Builds the CodeGPT-specific header set (auth + identity + routing hints)."""
    headers: dict[str, str] = {
        "tokens": "true",
        "source": "api",
        "channel": "api",
        "codegpt-version": config.CODEGPT_VERSION,
    }
    # Required by the inclusive-model endpoint; it names the model's upstream.
    if config.CODEGPT_PROVIDER:
        headers["X-Provider"] = config.CODEGPT_PROVIDER
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
            lines.append(f"[Earlier tool call: {name} with arguments {args}]")
    return "\n".join(lines)


def gemini_safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repairs a message list that Vertex/Gemini would reject with 400.

    The folded text is deliberately phrased as a record of past events rather
    than as a tool-call syntax, so the model does not learn to imitate it.

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
            pending.append(f"[Tool result: {content if content is not None else ''}]")
            continue
        if role in {"assistant", "model"}:
            calls = _flatten_tool_calls(msg.get("tool_calls"))
            text = msg.get("content")
            if isinstance(text, list):
                text = text_from_value(text)
            flush_pending()
            # Keep the assistant turn even when it only carried tool_calls;
            # dropping it leaves consecutive user turns, which Vertex rejects.
            out.append({"role": "assistant", "content": text or "(working)"})
            if calls:
                # Reported as incoming information rather than as the model's own
                # output: a model imitates its previous turns, so an emulated
                # call syntax would be parroted back to the user.
                pending.append(calls)
            continue
        flush_pending()
        text = msg.get("content")
        if isinstance(text, list):
            text = text_from_value(text)
        out.append({"role": role or "user", "content": text if text is not None else ""})
    flush_pending()

    # Collapse any run of same-role turns: Vertex treats them as one turn.
    collapsed: list[dict[str, Any]] = []
    for msg in out:
        if collapsed and collapsed[-1]["role"] == msg["role"]:
            merged = "\n".join(p for p in (str(collapsed[-1].get("content") or ""), str(msg.get("content") or "")) if p)
            collapsed[-1] = {"role": msg["role"], "content": merged}
        else:
            collapsed.append(msg)
    out = collapsed

    if not out:
        return [{"role": "user", "content": "(empty request)"}]
    while out and str(out[-1].get("role") or "").lower() in {"assistant", "model"}:
        out.append({"role": "user", "content": "Continue."})
    return out


# Models routinely invent tool names from their training data (`file_search`,
# `read_file`, `bash`). Auggie only knows its own names, so an invented call
# fails with "Tool X not found" and derails the turn.
#
# `file_search` is deliberately absent from this table: it genuinely can mean
# either a regex scan over files or a semantic lookup in Auggie's codebase
# index. Which one is meant is decided from the arguments, in `route_tool_call`.
_TOOL_ALIASES = {
    # file reading / listing
    "read": "view",
    "read_file": "view",
    "readfile": "view",
    "open_file": "view",
    "cat": "view",
    "view_file": "view",
    "get_file": "view",
    "list_files": "view",
    "list_dir": "view",
    "list_directory": "view",
    "ls": "view",
    "dir": "view",
    "glob": "view",
    "glob_files": "view",
    "find_files": "view",
    # writing / editing
    "write_file": "save-file",
    "write": "save-file",
    "create_file": "save-file",
    "save_file": "save-file",
    "edit_file": "str-replace-editor",
    "str_replace": "str-replace-editor",
    "strreplace": "str-replace-editor",
    "replace": "str-replace-editor",
    "patch": "apply_patch",
    "apply_patch": "apply_patch",
    "edit": "str-replace-editor",
    # shell / processes
    "bash": "launch-process",
    "shell": "launch-process",
    "sh": "launch-process",
    "run_command": "launch-process",
    "execute_command": "launch-process",
    "exec": "launch-process",
    "terminal": "launch-process",
    "run": "launch-process",
    # filesystem maintenance
    "delete_file": "remove-files",
    "delete_files": "remove-files",
    "rm": "remove-files",
    "remove_file": "remove-files",
    "remove_files": "remove-files",
    # web
    "web_search": "tavily_search_tavily",
    "search_web": "tavily_search_tavily",
    "fetch": "web-fetch",
    "fetch_url": "web-fetch",
    "http_get": "web-fetch",
    "browse": "web-fetch",
    # code intelligence / analysis (no direct Auggie equivalent: use the index)
    "audit_workspace": "view",
    "audit": "view",
    "analyze_workspace": "view",
    "analyze_repo": "view",
    "analyze_code": "view",
    "explore": "view",
    "explore_repo": "view",
    "understand_codebase": "view",
    "code_search": "launch-process",
    "search_code": "launch-process",
    "find_symbol": "view",
    "find_references": "view",
    "get_diagnostics": "view",
    # task tracking
    "todo_write": "add_tasks",
    "todowrite": "add_tasks",
    "create_tasks": "add_tasks",
    "update_tasks": "update_tasks",
    "list_tasks": "view_tasklist",
    "view_tasks": "view_tasklist",
}

# Keys that spell out a literal pattern mean a regex/glob scan, not a question.
_REGEX_KEYS = ("pattern", "regex", "regex_pattern", "search_query_regex", "glob", "query_regex")
# Keys that spell out a natural-language question mean semantic retrieval.
_SEMANTIC_KEYS = ("information_request", "information_need", "description", "question", "intent", "semantic_query")


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _looks_like_regex(value: str) -> bool:
    """A regex/glob hint: metacharacters that a plain question would not carry."""
    if not value:
        return False
    if any(ch in value for ch in ("*", "^", "$", "|", "\\", "(", ")", "[", "]")):
        return True
    return bool(re.search(r"\S+\.\w{1,5}\b|\w+\s*[:=]", value))


def shell_quote(value: str) -> str:
    """Single-quotes a value for safe interpolation into a shell command."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def route_tool_call(name: str, args: dict[str, Any], available: set[str]) -> tuple[str, dict[str, Any]]:
    """Chooses the real tool for an invented call and reshapes its arguments.

    `file_search` is the interesting case: a `pattern`/`glob` argument asks for a
    regex scan (Auggie's `view` with `search_query_regex`), while a plain question
    asks for the semantic index (`codebase-retrieval`). Guessing wrong either
    finds nothing or answers a different question, so the decision is made from
    the arguments rather than a fixed mapping.
    """
    lowered = name.lower()
    is_search = lowered in {"file_search", "search_files", "grep_search", "grep", "code_search", "search", "find_files", "find"}

    if is_search and name not in available:
        regex_hint = ""
        for key in _REGEX_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                regex_hint = value.strip()
                break
        # Also catch a pattern hidden under an unexpected key, as long as it is
        # not a natural-language question.
        if not regex_hint and not any(isinstance(args.get(k), str) and args.get(k, "").strip() for k in _SEMANTIC_KEYS):
            for value in args.values():
                if isinstance(value, str) and _looks_like_regex(value):
                    regex_hint = value.strip()
                    break
        semantic_present = any(isinstance(args.get(k), str) and args.get(k, "").strip() for k in _SEMANTIC_KEYS)
        target = args.get("path") or args.get("file") or args.get("file_path") or args.get("directory")
        # `view` can only regex-scan a single named file ("Optional parameter for
        # files only"); a directory or a bare pattern must go through the index.
        # Routing a recursive search at `view` made it fail on every turn, which
        # is what sent the model into a retry loop.
        file_level = isinstance(target, str) and target and not target.rstrip("/").endswith((".", "/"))

        if regex_hint and file_level and "view" in available and not semantic_present:
            shaped = {"path": target, "search_query_regex": regex_hint}
            if isinstance(args.get("case_sensitive"), bool):
                shaped["case_sensitive"] = args["case_sensitive"]
            return "view", shaped

        # A search must actually return matches. `view` only lists a directory,
        # which the model reads as "nothing found" and retries forever, so the
        # search is served by a real grep over the workspace.
        if "launch-process" in available:
            pattern = regex_hint or ""
            scope = target if isinstance(target, str) and target and target not in {".", "./"} else "."
            if pattern:
                command = f"grep -rn -- {shell_quote(pattern)} {shell_quote(scope)}"
            else:
                command = f"ls -la {shell_quote(scope)}"
            return "launch-process", {"command": command, "cwd": ".", "wait": True, "max_wait_seconds": 60}

        if "view" in available:
            shaped = {"path": ".", "type": "directory"}
            return "view", shaped

    mapped = _TOOL_ALIASES.get(lowered)
    if mapped and mapped in available:
        return mapped, args
    return resolve_tool_name(name, available), args


def resolve_tool_name(name: str, available: set[str]) -> str:
    """Maps a model-invented tool name onto a real Auggie tool, if one matches."""
    if name in available:
        return name
    mapped = _TOOL_ALIASES.get(name.lower())
    if mapped and mapped in available:
        return mapped
    # Last resort: an unambiguous suffix match (e.g. "view" vs "view-session").
    candidates = [t for t in available if t == name or t.endswith(f"_{name}") or t.startswith(f"{name}_")]
    if len(candidates) == 1:
        return candidates[0]
    return name


def _normalize_tool_calls(msg: dict[str, Any], available: set[str]) -> None:
    """Rewrites an assistant message's tool_call names (and arguments) in place."""
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        original = fn["name"]
        if original in available:
            continue
        args = _coerce_arguments(fn.get("arguments"))
        name, shaped = route_tool_call(original, args, available)
        fn["name"] = name
        if shaped != args:
            fn["arguments"] = json.dumps(shaped, ensure_ascii=False)


def adapt_request_body(request: dict[str, Any]) -> dict[str, Any]:
    """Rewrites an OpenAI chat body into the CodeGPT bridge request shape.

    OpenAI tools use {type, function:{name, description, parameters}}; CodeGPT
    expects a flat {name, description, parameters}. The model is addressed as
    `modelId` (not `model`) and a `session_id` accompanies `requestId`; no agent
    is involved.
    """
    raw_tools = request.get("tools") if isinstance(request.get("tools"), list) else []
    available = {
        (t.get("function") or {}).get("name") or t.get("name")
        for t in raw_tools
        if isinstance(t, dict)
    }
    available = {n for n in available if isinstance(n, str)}
    messages = [dict(m) for m in (request.get("messages") or []) if isinstance(m, dict)]
    for msg in messages:
        _normalize_tool_calls(msg, available)

    body: dict[str, Any] = {
        "messages": gemini_safe_messages(messages),
        "temperature": request.get("temperature", 0),
        "stream": True,
        "format": "json",
        "requestId": str(uuid.uuid4()),
        "session_id": config.CODEGPT_SESSION_ID or f"{config.CODEGPT_HARNESS}-{session_token()[:24]}",
    }
    model_id = request.get("model") or config.TARGET_MODEL
    if isinstance(model_id, str) and model_id:
        body["modelId"] = model_id

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
