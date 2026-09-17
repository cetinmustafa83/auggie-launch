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
# Dynamic Model Registry & 9router Catalog Discovery
# ============================================================================

def fetch_upstream_models() -> list[dict[str, Any]]:
    """Fetches real model list from 9router / upstream /v1/models endpoint."""
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

    9router exposes one; CodeGPT Plus exposes another. The CodeGPT catalog is
    the only place deepseek's 1M window is stated, and missing it made the proxy
    inject a 200k limit -- so history compacted far earlier than necessary.
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
        entry = config.CACHED_CATALOG.get(model_id)
        if isinstance(entry, dict):
            ctx = entry.get("contextWindow") or entry.get("context_window") or entry.get("contextLength")
            if isinstance(ctx, (int, float)) and ctx > 0:
                return int(ctx)
    # Last segment match (e.g., "gpt-4o" from "openai/gpt-4o")
    if "/" in model_id:
        last_segment = model_id.split("/")[-1]
        if last_segment in config.CACHED_CATALOG:
            entry = config.CACHED_CATALOG.get(last_segment)
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
        return config.MODEL_CONTEXT_TOKENS if config.MODEL_CONTEXT_TOKENS > 0 else 200000
    return min(limits)


def effective_context_limit(model_id: str) -> int:
    """Context budget actually used for truncation and injection for one model.

    An explicit AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS always wins (operator override);
    otherwise combos use their weakest member and everything else uses the catalog.
    """
    if config.MODEL_CONTEXT_TOKENS_EXPLICIT and config.MODEL_CONTEXT_TOKENS > 0:
        return config.MODEL_CONTEXT_TOKENS
    for combo in config._LOCAL_9ROUTER.combos:
        if combo.get("name") == model_id:
            return combo_context_limit(combo)
    alias_target = config._LOCAL_9ROUTER.model_aliases.get(model_id)
    if alias_target:
        aliased = model_context_limit(alias_target)
        if aliased > 0:
            return aliased
    return model_context_limit(model_id)


def known_model_names() -> set[str]:
    """Every model name the launcher advertises to Auggie."""
    names = {config.TARGET_MODEL}
    names.update(str(c.get("name")) for c in config._LOCAL_9ROUTER.combos if c.get("name"))
    names.update(config._LOCAL_9ROUTER.model_aliases.keys())
    names.update(
        str(item.get("id")) for item in config._CACHED_MODELS if isinstance(item, dict) and item.get("id")
    )
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
    return config.MODEL_CONTEXT_TOKENS


