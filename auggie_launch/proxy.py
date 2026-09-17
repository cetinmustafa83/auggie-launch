from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler
from typing import Any

from . import config
from .config import log
from .models import effective_context_limit
from .registry import fake_batch_upload, fake_find_missing, fake_generic, fake_models, fake_token
from .transform import build_openai_request, text_from_value
from .truncation import tool_calls_to_nodes
from .upstream import compact_upstream_error, json_bytes, open_upstream_with_retries

# ============================================================================
# Local Auggie Proxy Server
# ============================================================================

def dump_debug_request(body: Any, openai_request: dict[str, Any]) -> None:
    """Write incoming/outgoing payloads to the debug dir when verbose mode is on.

    A high-volume debug dir is not free: without a bound it grows one pair of
    files per request and is never cleaned up. Only the most recent dumps are
    kept, and the old ones are pruned here rather than left to the user.
    """
    if not config.VERBOSE:
        return
    os.makedirs(config.DEBUG_DIR, exist_ok=True)
    for name, value in (("incoming_augment_request.json", body), ("outgoing_openai_request.json", openai_request)):
        with open(os.path.join(config.DEBUG_DIR, name), "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    prune_debug_dir()


def prune_debug_dir(keep_seconds: float | None = None) -> int:
    """Deletes debug dumps older than the retention window. Returns the count."""
    window = config.DEBUG_RETENTION_SECONDS if keep_seconds is None else keep_seconds
    directory = config.DEBUG_DIR
    if not directory or not os.path.isdir(directory):
        return 0
    cutoff = time.time() - max(0.0, window)
    removed = 0
    try:
        entries = os.listdir(directory)
    except OSError:
        return 0
    for entry in entries:
        path = os.path.join(directory, entry)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            continue
    return removed


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
                if reasoning and config.STREAM_THINKING and not content.startswith("<think>"):
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
    context_tokens = effective_context_limit(request_model) if isinstance(request_model, str) else config.MODEL_CONTEXT_TOKENS
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
        "max_context_tokens": context_tokens or config.MODEL_CONTEXT_TOKENS,
        "max_output_tokens": config.MODEL_MAX_OUTPUT_TOKENS,
    }


def resolve_tool_calls(openai_request: dict[str, Any], tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrites tool names the model invented onto the tools Auggie actually has.

    Streaming deltas are forwarded as they arrive, but the model may emit an
    invented name (e.g. `grep_search`). Auggie would answer "Tool not found" and
    the turn would be wasted, so names are resolved before they reach it.
    """
    if not config.IS_CODEGPT or not tool_calls:
        return tool_calls
    from . import codegpt

    available: set[str] = set()
    for tool in openai_request.get("tools") or []:
        if isinstance(tool, dict):
            fn_raw = tool.get("function")
            fn: dict[str, Any] = fn_raw if isinstance(fn_raw, dict) else tool
            name = fn.get("name")
            if isinstance(name, str):
                available.add(name)
    if not available:
        return tool_calls

    from .truncation import merge_stream_tool_calls

    # Merge the streamed fragments FIRST. Remapping each delta individually
    # would rewrite a partial (often empty) argument set into a full command and
    # then concatenate that with the real arguments, producing invalid JSON.
    resolved: list[dict[str, Any]] = []
    for call in merge_stream_tool_calls(tool_calls):
        fn_src = call.get("function")
        fn_call: dict[str, Any] = dict(fn_src) if isinstance(fn_src, dict) else {}
        if isinstance(fn_call.get("name"), str):
            args = codegpt._coerce_arguments(fn_call.get("arguments"))
            name, shaped = codegpt.route_tool_call(fn_call["name"], args, available)
            if name != fn_call["name"]:
                log(f"tool remap: {fn_call['name']} -> {name}")
                from . import stats as stats_mod
                stats_mod.record_remap(str(fn_call['name']), name)
            fn_call["name"] = name
            if shaped != args:
                fn_call["arguments"] = json.dumps(shaped, ensure_ascii=False)
            call = {**call, "function": fn_call}
        resolved.append(call)
    return resolved


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
    server_version = f"auggie-launch/{config.__version__} (codegpt-native)"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if config.VERBOSE:
            super().log_message(fmt, *args)

    def handle_one_request(self) -> None:
        """Handles a request, tolerating a client that goes away mid-request.

        A reset is normal here: the CLI abandons in-flight requests when it is
        interrupted, and socketserver would otherwise print a full traceback for
        something that is not a fault.
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            self.close_connection = True

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
        if not config.REQUIRE_LOCAL_TOKEN:
            return True
        if path in {"", "health"} or path in {"token", "auth/token"}:
            return True
        header = self.headers.get("Authorization") or self.headers.get("authorization") or ""
        presented = header[7:].strip() if header.lower().startswith("bearer ") else header.strip()
        if not presented:
            presented = (self.headers.get("X-Api-Key") or "").strip()
        if presented and secrets.compare_digest(presented, config.LOCAL_TOKEN):
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
                "version": config.__version__,
                "model": config.TARGET_MODEL,
                "indexing_mode": config.INDEXING_MODE,
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
        if config.VERBOSE:
            log(f"{self.command} /{path}")

        if path in {"token", "auth/token"}:
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
        dump_debug_request(body, openai_request)
        started = time.time()
        from . import stats as stats_mod
        stats_mod.record_request("json", len(json_bytes(openai_request)))
        try:
            with open_upstream_with_retries(json_bytes(openai_request), stream=False, timeout=int(config.UPSTREAM_TIMEOUT_SECONDS), label="json") as resp:
                raw = resp.read()
            data = json.loads(raw.decode("utf-8") or "{}")
            text = extract_chat_text(data)
            tool_calls: list[dict[str, Any]] = []
            choices = data.get("choices") if isinstance(data, dict) else None
            if isinstance(choices, list) and choices:
                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
                    tool_calls = [call for call in message["tool_calls"] if isinstance(call, dict)]
            from . import stats as stats_mod
            stats_mod.record_latency(time.time() - started)
            for call in tool_calls:
                fn = call.get("function") if isinstance(call, dict) else None
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    stats_mod.record_tool_call(fn["name"])
            usage_value = data.get("usage") if isinstance(data, dict) else None
            stats_mod.record_usage(
                usage_value if isinstance(usage_value, dict) else estimate_usage(openai_request, None),
                estimated=not isinstance(usage_value, dict),
            )
            self.send_json(augment_chat_response(text, str(uuid.uuid4()), openai_request, usage_value, tool_calls))
        except urllib.error.HTTPError as exc:
            raw = compact_upstream_error(exc.msg, max_chars=2000)
            from . import stats as stats_mod
            stats_mod.record_error(f"HTTP {exc.code}")
            self.send_json({"error": "upstream_error", "message": raw, "status": exc.code}, status=502 if exc.code in {401, 403} else exc.code)
        except Exception as exc:
            self.send_json({"error": "upstream_error", "message": compact_upstream_error(str(exc))}, status=502)

    def forward_stream(self, body: Any) -> None:
        openai_request = build_openai_request(body, stream=True)
        dump_debug_request(body, openai_request)
        started = time.time()
        from . import stats as stats_mod
        stats_mod.record_request("stream", len(json_bytes(openai_request)))
        if config.VERBOSE:
            log(f"stream payload: {len(json_bytes(openai_request))} bytes, {len(openai_request.get('messages', []))} messages, {len(openai_request.get('tools') or [])} tools")
        request_id = str(uuid.uuid4())
        try:
            upstream_wrapper = open_upstream_with_retries(json_bytes(openai_request), stream=True, timeout=int(config.UPSTREAM_TIMEOUT_SECONDS), label="stream")
        except urllib.error.HTTPError as exc:
            msg = compact_upstream_error(exc.msg, max_chars=2000)
            from . import stats as stats_mod
            stats_mod.record_error(f"HTTP {exc.code}")
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
                        if config.STREAM_THINKING:
                            if not in_thinking_block:
                                in_thinking_block = True
                                write_chunk({"delta": "<think>\n", "request_id": request_id})
                                accumulated.append("<think>\n")
                            accumulated.append(reasoning_text)
                            write_chunk({"delta": reasoning_text, "request_id": request_id})
                        else:
                            write_chunk({"text": "", "heartbeat": True, "thinking": True, "request_id": request_id})

                    if delta_text:
                        if in_thinking_block:
                            in_thinking_block = False
                            write_chunk({"delta": "\n</think>\n\n", "request_id": request_id})
                            accumulated.append("\n</think>\n\n")

                        accumulated.append(delta_text)
                        write_chunk({"delta": delta_text, "request_id": request_id})

        except Exception as exc:
            if not client_disconnected:
                from . import stats as stats_mod
                stats_mod.record_error(type(exc).__name__)
                write_chunk({"error": "upstream_error", "message": compact_upstream_error(str(exc)), "request_id": request_id})
        finally:
            if in_thinking_block and not client_disconnected:
                write_chunk({"delta": "\n</think>\n\n", "request_id": request_id})
                accumulated.append("\n</think>\n\n")

            if not client_disconnected:
                from . import stats as stats_mod
                stats_mod.record_latency(time.time() - started)
                # CodeGPT sends no usage object, so fall back to the same
                # estimate used for the Augment-facing token_usage, flagged as such.
                stats_mod.record_usage(
                    usage if isinstance(usage, dict) else estimate_usage(openai_request, None),
                    estimated=not isinstance(usage, dict),
                )
                for call in tool_calls:
                    fn = call.get("function") if isinstance(call, dict) else None
                    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                        stats_mod.record_tool_call(fn["name"])
                final_text = "".join(accumulated)
                merged_calls = resolve_tool_calls(openai_request, tool_calls)
                final = augment_chat_response(final_text, request_id, openai_request, usage, merged_calls)
                final["done"] = True
                write_chunk(final)
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass


