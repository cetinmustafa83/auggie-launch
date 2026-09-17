from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from . import config
from .config import log

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

    reserve = max(config.MODEL_MAX_OUTPUT_TOKENS, 4096) + 1024
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
    """Deterministically merges streaming delta tool calls by index and id.

    Deltas for the same call share an index and accumulate argument fragments.
    Some upstreams (CodeGPT) restart the index at 0 for every parallel call, so
    a changing `id` also starts a new call -- otherwise two calls' arguments get
    concatenated into invalid JSON like `{"a":1}{"b":2}`.
    """
    merged: list[dict[str, Any]] = []
    by_index: dict[Any, dict[str, Any]] = {}

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        idx = call.get("index")
        if idx is None:
            idx = len(merged)
        call_id = call.get("id")
        fn_raw = call.get("function")
        fn: dict[str, Any] = fn_raw if isinstance(fn_raw, dict) else {}

        target = by_index.get(idx)
        # A new id on a seen index means a fresh parallel call, not a continuation.
        if target is not None and call_id and target.get("id") and call_id != target["id"]:
            target = None
        if target is None:
            target = {
                "id": call_id or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
            merged.append(target)
            by_index[idx] = target
        elif call_id:
            target["id"] = call_id
        fn_target = target["function"]
        if isinstance(fn.get("name"), str):
            fn_target["name"] = str(fn_target.get("name") or "") + fn["name"]
        if isinstance(fn.get("arguments"), str):
            fn_target["arguments"] = str(fn_target.get("arguments") or "") + fn["arguments"]

    result: list[dict[str, Any]] = []
    for item in merged:
        fn_item = item["function"]
        name = str(fn_item.get("name") or "").strip()
        if not name:
            continue
        fn_item["arguments"] = repair_json_arguments(str(fn_item.get("arguments") or ""))
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


