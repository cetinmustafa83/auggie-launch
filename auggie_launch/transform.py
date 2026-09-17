from __future__ import annotations

import json
import os
import uuid
from typing import Any

from . import config
from .models import effective_context_limit, resolve_request_model
from .truncation import truncate_messages_to_context_limit

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
    """Combines the custom prompt with the language rule."""
    parts: list[str] = []
    base_prompt = os.environ.get("AUGGIE_LAUNCH_SYSTEM_PROMPT", "").strip()
    if base_prompt:
        parts.append(base_prompt)

    # Upstreams otherwise drift into an arbitrary language mid-answer (seen with
    # the CodeGPT agent). Anchoring it keeps replies in the user's language.
    language = (config.REPLY_LANGUAGE or "").strip()
    if language and language.lower() not in {"keep", "auto", "same"}:
        parts.append(
            f"Always write your answers in {language}. "
            "Use the same language the user writes in unless told otherwise."
        )


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
    if config.USE_COMPLETION_TOKENS == "true":
        return True
    if config.USE_COMPLETION_TOKENS == "false":
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
    output_ceiling = max(256, min(config.MODEL_MAX_OUTPUT_TOKENS, max(1024, limit // 4)))

    if isinstance(body, dict):
        if isinstance(body.get("temperature"), (int, float)):
            request["temperature"] = body["temperature"]
        max_tokens = body.get("max_tokens") or body.get("max_output_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            request[tokens_field] = min(max_tokens, output_ceiling)
        if isinstance(body.get("reasoning_effort"), str) and body["reasoning_effort"].strip():
            request["reasoning_effort"] = body["reasoning_effort"].strip().lower()

    if "reasoning_effort" not in request and config.REASONING_EFFORT:
        request["reasoning_effort"] = config.REASONING_EFFORT

    return request


