from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from . import config
from .config import log
from .upstream import _CONNECTION_POOL, active_base_url, get_active_key, upstream_headers

# ============================================================================
# Dynamic Model Registry & Catalog Discovery
# ============================================================================

def fetch_upstream_models() -> list[dict[str, Any]]:
    """Fetches the real model list from the upstream /models endpoint."""
    now = time.time()
    with config._MODELS_LOCK:
        if config._CACHED_MODELS and (now - config._CACHED_MODELS_TIME) < config._MODELS_CACHE_TTL:
            return config._CACHED_MODELS

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
                with config._MODELS_LOCK:
                    config._CACHED_MODELS = parsed_models
                    config._CACHED_MODELS_TIME = now
                log(f"loaded {len(parsed_models)} dynamic models from upstream registry")
                _CONNECTION_POOL.release(parsed_url, conn, reusable=True)
                return parsed_models
    except Exception as exc:
        log(f"could not fetch dynamic models: {exc}")
    finally:
        _CONNECTION_POOL.release(parsed_url, conn, reusable=False)

    return config._CACHED_MODELS


def lookup_catalog_context(model_id: str) -> int:
    """Context window for a model, from whichever catalog is in play.

    The CodeGPT catalog is the only place deepseek's 1M window is stated, and
    missing it made the proxy inject a 200k limit -- so history compacted far
    earlier than necessary.
    """
    if config.IS_CODEGPT:
        try:
            from . import codegpt
            wanted = (model_id or "").split("/")[-1].strip().lower()
            for entry in codegpt.load_catalog_models():
                if entry["id"].lower() == wanted:
                    return int(entry.get("context") or 0)
        except Exception:
            pass
    if not config.CACHED_CATALOG:
        return 0
    # Exact model ID match
    if model_id in config.CACHED_CATALOG:
        entry1: Any = config.CACHED_CATALOG.get(model_id)
        if isinstance(entry1, dict):
            ctx = entry1.get("contextWindow") or entry1.get("context_window") or entry1.get("contextLength")
            if isinstance(ctx, (int, float)) and ctx > 0:
                return int(ctx)
    # Last segment match (e.g., "gpt-4o" from "openai/gpt-4o")
    if "/" in model_id:
        last_segment = model_id.split("/")[-1]
        if last_segment in config.CACHED_CATALOG:
            entry2: Any = config.CACHED_CATALOG.get(last_segment)
            if isinstance(entry2, dict):
                ctx2 = entry2.get("contextWindow") or entry2.get("context_window") or entry2.get("contextLength")
                if isinstance(ctx2, (int, float)) and ctx2 > 0:
                    return int(ctx2)
    return 0

def effective_context_limit(model_id: str) -> int:
    """Context budget actually used for truncation and injection for one model.

    An explicit AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS always wins (operator override);
    otherwise the catalog decides.
    """
    if config.MODEL_CONTEXT_TOKENS_EXPLICIT and config.MODEL_CONTEXT_TOKENS > 0:
        return config.MODEL_CONTEXT_TOKENS
    return model_context_limit(model_id)


def known_model_names() -> set[str]:
    """Every model name the launcher advertises to the CLI.

    The cached catalog matters as much as the live model list: it is where the
    per-model context windows live, so a model missing from here silently falls
    back to the default window and gets truncated too early.
    """
    names = {config.TARGET_MODEL}
    names.update(str(key) for key in config.CACHED_CATALOG)
    names.update(
        str(item.get("id")) for item in config._CACHED_MODELS if isinstance(item, dict) and item.get("id")
    )
    if config.IS_CODEGPT:
        try:
            from . import codegpt
            names.update(entry["id"] for entry in codegpt.load_catalog_models())
        except Exception:
            pass
    return names


def resolve_request_model(body: Any) -> str:
    """Honours the model Auggie asked for when we advertise it, else the configured target."""
    if not isinstance(body, dict):
        return config.TARGET_MODEL
    for key in ("model", "model_name", "internal_name", "modelName"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            if candidate in known_model_names():
                return candidate
            log(f"ignoring unknown requested model '{candidate}', using {config.TARGET_MODEL}")
            break
    return config.TARGET_MODEL


def model_context_limit(model_id: str) -> int:
    """Heuristic context window for a model, with catalog priority."""
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
    return config.MODEL_CONTEXT_TOKENS


