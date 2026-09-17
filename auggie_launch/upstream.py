from __future__ import annotations

import http.client
import json
import random
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any

from . import config
from .config import log, with_codex_headers

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
                    # A pooled connection keeps the timeout it was created with;
                    # refresh it so a later, longer-lived request is not cut off
                    # by an earlier short timeout (and vice versa).
                    try:
                        if conn.sock is not None:
                            conn.sock.settimeout(timeout)
                        conn.timeout = timeout
                    except Exception:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        continue
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
    if config.UPSTREAM_USER_AGENT:
        headers["User-Agent"] = config.UPSTREAM_USER_AGENT
    # CodeGPT Plus: identity/routing headers and a token minted by our own bridge.
    if config.IS_CODEGPT:
        from . import codegpt  # local import: codegpt imports config at module load
        headers.update(codegpt.extra_headers())


    return with_codex_headers(headers)


def active_base_url() -> str:
    return config.ACTIVE_BASE_URL or config.TARGET_BASE_URL


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


def upstream_url(has_tools: bool = False) -> str:
    if config.IS_CODEGPT:
        from . import codegpt
        return f"{active_base_url()}{codegpt.chat_path(has_tools)}"
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
    return min(max(0.0, selected), config.UPSTREAM_MAX_RETRY_AFTER_SECONDS)


def retry_backoff_seconds(attempt: int) -> float:
    """Full jitter exponential backoff (AWS architecture standard)."""
    base = min(config.UPSTREAM_BACKOFF_MAX_SECONDS, config.UPSTREAM_BACKOFF_INITIAL_SECONDS * (2 ** max(0, attempt)))
    return random.uniform(0.0, base)


def apply_upstream_cooldown(seconds: float, reason: str) -> None:
    if seconds <= 0:
        return
    capped = min(seconds, config.UPSTREAM_MAX_RETRY_AFTER_SECONDS)
    with config.UPSTREAM_THROTTLE_LOCK:
        until = time.time() + capped
        if until > config.UPSTREAM_COOLDOWN_UNTIL:
            config.UPSTREAM_COOLDOWN_UNTIL = until
    log(f"upstream cooldown {capped:.2f}s ({reason})")


def wait_for_upstream_slot() -> None:
    while True:
        with config.UPSTREAM_THROTTLE_LOCK:
            now = time.time()
            wait_until = max(config.UPSTREAM_COOLDOWN_UNTIL, config.UPSTREAM_NEXT_REQUEST_AT)
            wait_for = wait_until - now
            if wait_for <= 0:
                config.UPSTREAM_NEXT_REQUEST_AT = now + config.UPSTREAM_MIN_INTERVAL_SECONDS
                return
        time.sleep(min(wait_for, 5.0))


def reset_throttles() -> None:
    """Resets all throttling and cooldown timers."""
    with config.UPSTREAM_THROTTLE_LOCK:
        config.UPSTREAM_NEXT_REQUEST_AT = 0.0
        config.UPSTREAM_COOLDOWN_UNTIL = 0.0


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
    if config.IS_CODEGPT:
        from . import codegpt
        token = codegpt.session_token()
        if token:
            return token
    with config.FROZEN_LOCK:
        now = time.time()
        for key in config.API_KEYS:
            if config.FROZEN_KEYS.get(key, 0) <= now:
                return key
        return config.API_KEYS[0] if config.API_KEYS else ""


def mark_key_failed(key: str, status_code: int, retry_after: str | None = None) -> float:
    with config.FROZEN_LOCK:
        now = time.time()
        if status_code == 429:
            freeze_for = clamp_retry_delay(parse_retry_after(retry_after, now=now), config.UPSTREAM_429_FREEZE_SECONDS)
            config.FROZEN_KEYS[key] = now + freeze_for
            return freeze_for
        elif status_code in (401, 402):
            config.FROZEN_KEYS[key] = now + 86400
            return 86400.0
        elif 500 <= status_code <= 599:
            freeze_for = config.UPSTREAM_5XX_FREEZE_SECONDS
            config.FROZEN_KEYS[key] = now + freeze_for
            return freeze_for
    return 0.0


def log_router_response_headers(headers: dict[str, str] | http.client.HTTPMessage) -> None:
    """Logs upstream routing metadata in verbose mode."""
    if not config.VERBOSE:
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
        log(f"upstream: {' | '.join(details)}")


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


def codegpt_has_tools(data: bytes) -> bool:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        return False
    tools = payload.get("tools") if isinstance(payload, dict) else None
    return isinstance(tools, list) and bool(tools)


def open_upstream_with_retries(data: bytes, *, stream: bool, timeout: int, label: str) -> UpstreamResponseWrapper:
    """Executes upstream HTTP requests with Keep-Alive pool, full jitter, and parameter negotiation."""
    if config.IS_CODEGPT:
        from . import codegpt
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            data = json_bytes(codegpt.adapt_request_body(payload))

    has_tools = codegpt_has_tools(data) if config.IS_CODEGPT else False
    parsed_url = urllib.parse.urlparse(upstream_url(has_tools))
    path_with_query = parsed_url.path or "/chat/completions"
    if parsed_url.query:
        path_with_query += f"?{parsed_url.query}"

    last_error: Exception | None = None
    max_attempts = max(len(config.API_KEYS), 1) + max(0, config.UPSTREAM_RETRIES)
    if config.IS_CODEGPT:
        # A single session token, so retries only help against transient errors.
        max_attempts = max(1, config.UPSTREAM_RETRIES + 1)
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
                    min(frozen_for or config.UPSTREAM_429_FREEZE_SECONDS, retry_backoff_seconds(attempt)),
                )
                apply_upstream_cooldown(cooldown, "429 rate limit")
            elif 500 <= response.status <= 599:
                apply_upstream_cooldown(retry_backoff_seconds(attempt), f"{response.status} upstream error")

            _CONNECTION_POOL.release(parsed_url, conn, reusable=False)

            if is_retryable_upstream_status(response.status) and attempt < max_attempts - 1:
                continue
            if response.status in (401, 402) and attempt < min(len(config.API_KEYS), max_attempts) - 1:
                continue
            break

        except Exception as exc:
            last_error = exc
            _CONNECTION_POOL.release(parsed_url, conn, reusable=False)
            if attempt < max_attempts - 1:
                apply_upstream_cooldown(retry_backoff_seconds(attempt), "transport error")
                continue
            break

    if last_error is not None:
        raise last_error
    raise RuntimeError("upstream request failed")


