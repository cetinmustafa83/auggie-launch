from __future__ import annotations

import json
import os
import statistics
import tempfile
import threading
import time
from typing import Any

# ============================================================================
# Usage counters
# ============================================================================
#
# Nothing recorded what this proxy actually did, so "is it slow?", "which tool
# does the model reach for?", "how often does it invent a name?" were answered by
# reading logs by hand. These are counters, not a trace: small, bounded, and
# written on a coarser cadence than the requests they describe.

_LOCK = threading.Lock()
_STATE: dict[str, Any] | None = None
_dirty = False
_last_flush = 0.0
_FLUSH_INTERVAL = 5.0


def stats_path() -> str:
    override = (os.environ.get("AUGGIE_LAUNCH_STATS_PATH") or "").strip()
    if override:
        return override
    return os.path.join(tempfile.gettempdir(), "auggie-launch.stats.json")


def _empty() -> dict[str, Any]:
    return {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requests": 0,
        "streams": 0,
        "json_requests": 0,
        "errors": 0,
        "tool_calls": 0,
        "tool_names": {},
        "remaps": {},
        "payload_bytes": [],
        "latency_ms": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "usage_estimated": False,
        "retries": 0,
    }


def _load() -> dict[str, Any]:
    global _STATE
    if _STATE is not None:
        return _STATE
    path = stats_path()
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        _STATE = loaded if isinstance(loaded, dict) else _empty()
    except Exception:
        _STATE = _empty()
    return _STATE


def _flush_locked() -> None:
    state = _STATE
    if not state:
        return
    path = stats_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        # Counters are diagnostics; failing to write them must never fail a request.
        pass


def _flush(force: bool = False) -> None:
    global _dirty, _last_flush
    now = time.time()
    if not _dirty:
        return
    if not force and (now - _last_flush) < _FLUSH_INTERVAL:
        return
    _flush_locked()
    _dirty = False
    _last_flush = now


def _bump(counter: str, amount: int = 1) -> None:
    state = _load()
    state[counter] = int(state.get(counter) or 0) + amount
    global _dirty
    _dirty = True


def _bump_mapping(counter: str, key: str) -> None:
    if not key:
        return
    state = _load()
    bucket = state.setdefault(counter, {})
    if not isinstance(bucket, dict):
        bucket = {}
        state[counter] = bucket
    bucket[key] = int(bucket.get(key) or 0) + 1
    global _dirty
    _dirty = True


def record_request(kind: str, payload_bytes: int = 0) -> None:
    """Counts one request. `kind` is 'stream' or 'json'."""
    with _LOCK:
        _bump("requests")
        _bump("streams" if kind == "stream" else "json_requests")
        if payload_bytes > 0:
            sizes = _load().setdefault("payload_bytes", [])
            if isinstance(sizes, list):
                sizes.append(int(payload_bytes))
                del sizes[:-200]  # bounded: the trend matters, not the history
        _flush()


def record_latency(seconds: float) -> None:
    with _LOCK:
        samples = _load().setdefault("latency_ms", [])
        if isinstance(samples, list):
            samples.append(int(max(0.0, seconds) * 1000))
            del samples[:-200]
        global _dirty
        _dirty = True
        _flush()


def record_error(message: str = "") -> None:
    with _LOCK:
        _bump("errors")
        if message:
            _bump_mapping("remaps", f"error: {message[:40]}")
        _flush()


def record_retry(reason: str = "") -> None:
    """Counts an upstream retry, so backoff is visible rather than silent."""
    with _LOCK:
        _bump("retries")
        if reason:
            _bump_mapping("retries_by_reason", reason[:32])
        _flush()


def record_tool_call(name: str) -> None:
    with _LOCK:
        _bump("tool_calls")
        _bump_mapping("tool_names", name)
        _flush()


def record_remap(source: str, target: str) -> None:
    with _LOCK:
        _bump_mapping("remaps", f"{source} -> {target}")
        _flush()


def record_usage(usage: Any, estimated: bool = False) -> None:
    """Adds token counts.

    CodeGPT's stream carries no usage object at all, so the proxy's own estimate
    is what there is. `estimated` records that, so the numbers are not read as
    billing-accurate when they are a character-count approximation.
    """
    if not isinstance(usage, dict):
        return
    with _LOCK:
        state = _load()
        state["input_tokens"] = int(state.get("input_tokens") or 0) + int(usage.get("prompt_tokens") or 0)
        state["output_tokens"] = int(state.get("output_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        if estimated:
            state["usage_estimated"] = True
        global _dirty
        _dirty = True
        _flush()


def snapshot() -> dict[str, Any]:
    with _LOCK:
        state = dict(_load())
        _flush(force=True)
    return state


def reset() -> None:
    global _STATE, _dirty
    with _LOCK:
        _STATE = _empty()
        _dirty = True
        _flush(force=True)


def _top(mapping: Any, limit: int = 5) -> list[tuple[str, int]]:
    if not isinstance(mapping, dict):
        return []
    rows = sorted(((str(k), int(v)) for k, v in mapping.items()), key=lambda r: r[1], reverse=True)
    return rows[:limit]


def format_stats(state: dict[str, Any] | None = None) -> str:
    """Renders the counters. Returns the text so it can be tested directly."""
    data = state if state is not None else snapshot()
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("auggie-launch usage")
    lines.append("=" * 62)
    lines.append(f"  since            {data.get('started', '?')}  ({stats_path()})")
    lines.append(f"  requests         {data.get('requests', 0)}  "
                 f"({data.get('streams', 0)} stream, {data.get('json_requests', 0)} json)")
    lines.append(f"  errors           {data.get('errors', 0)}")

    tokens_in = int(data.get("input_tokens") or 0)
    tokens_out = int(data.get("output_tokens") or 0)
    if tokens_in or tokens_out:
        suffix = " (estimated: the stream carries no usage object)" if data.get("usage_estimated") else ""
        lines.append(f"  tokens in/out    {tokens_in:,} / {tokens_out:,}{suffix}")

    sizes = [int(s) for s in (data.get("payload_bytes") or []) if isinstance(s, (int, float))]
    if sizes:
        lines.append(f"  payload          avg {sum(sizes) // len(sizes):,} B, "
                     f"largest {max(sizes):,} B  (last {len(sizes)})")

    latencies = [int(x) for x in (data.get("latency_ms") or []) if isinstance(x, (int, float))]
    if latencies:
        lines.append(f"  latency          p50 {int(statistics.median(latencies))} ms, "
                     f"max {max(latencies)} ms  (last {len(latencies)})")

    if int(data.get("retries") or 0):
        lines.append(f"  retries          {data['retries']}")
    tools = _top(data.get("tool_names"))
    if tools:
        lines.append("  tool calls       " + ", ".join(f"{name} ({count})" for name, count in tools))

    remaps = _top(data.get("remaps"))
    if remaps:
        lines.append("  remaps           " + ", ".join(f"{name} ({count})" for name, count in remaps))

    if int(data.get("requests") or 0) == 0:
        lines.append("")
        lines.append("  no requests recorded yet")
    return "\n".join(lines)


def print_stats() -> None:
    print(format_stats())
