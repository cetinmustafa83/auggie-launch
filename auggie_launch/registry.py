from __future__ import annotations

import json
import uuid
from typing import Any

from . import config
from .models import combo_context_limit, effective_context_limit, fetch_upstream_models

# ============================================================================
# Full Model Registry & Session Injection
# ============================================================================

def fake_token() -> dict[str, Any]:
    return {
        "access_token": config.LOCAL_TOKEN,
        "token_type": "Bearer",
        "expires_in": 31536000,
        "scope": "email profile offline_access",
    }


def model_list_entry(model_id: str, context_tokens: int) -> dict[str, Any]:
    """Auggie model descriptor whose completion budgets track the real context window.

    Auggie sizes its prefix/suffix payloads from these counts, so deriving them from
    the model's own window is what keeps large-context models usable and small ones
    from overflowing upstream.
    """
    context = context_tokens if context_tokens > 0 else 200000
    # ~4 chars/token: one quarter of the window per side leaves half the window
    # for history, tool definitions and the response.
    half_budget_chars = max(2000, min(200000, context))
    return {
        "name": model_id,
        "internal_name": model_id,
        "suggested_prefix_char_count": half_budget_chars,
        "suggested_suffix_char_count": half_budget_chars,
        # Keep Auggie's own deadline under the proxy's upstream timeout, so it
        # reports a clean timeout instead of a broken stream when we cut first.
        "completion_timeout_ms": int(max(30.0, config.UPSTREAM_TIMEOUT_SECONDS - 30.0) * 1000),
    }


def model_registry_entry(model_id: str, *, description: str, group: str = "") -> dict[str, Any]:
    """Descriptor for Auggie's model picker.

    Auggie reads `displayName`/`shortName` from the registry when building the
    `/model` menu (see its `nte()`/resolver helpers); without them the menu
    crashes while rendering. `modelGroup` drives the picker's grouping label.
    """
    entry: dict[str, Any] = {
        "humanName": model_id,
        "displayName": model_id,
        "shortName": model_id,
        "description": description,
        "encoding": "o200k_base",
        "isRecommended": True,
    }
    if group:
        entry["modelGroup"] = group
    return entry

def history_summary_params() -> str:
    """JSON string for Auggie's `history_summary_params` feature flag.

    `trigger_on_total_tokens` makes the CLI compact the session before the
    upstream window is exhausted; `max_history_chars` bounds the abridged tail
    that is kept verbatim. Auggie parses this as JSON and expects snake_case.
    """
    # The trigger is a share of the model's real window, so a 1M-token model is
    # not compacted as if it only had 200k. An explicit env value still wins.
    trigger = config.HISTORY_SUMMARY_TRIGGER_TOKENS
    if not config.HISTORY_SUMMARY_TRIGGER_EXPLICIT:
        window = effective_context_limit(config.TARGET_MODEL)
        if window > 0:
            trigger = max(16000, int(window * config.HISTORY_SUMMARY_TRIGGER_RATIO))
    # The verbatim tail kept alongside the summary should also scale, otherwise
    # a large-window model still loses most of its recent context.
    keep_chars = config.HISTORY_SUMMARY_MAX_HISTORY_CHARS
    if not config.HISTORY_SUMMARY_MAX_HISTORY_EXPLICIT:
        window = effective_context_limit(config.TARGET_MODEL)
        if window > 0:
            # ~10% of the window at ~4 chars/token, capped so the summary does
            # not become pointless (keeping nearly everything verbatim).
            keep_chars = max(40000, min(400000, int(window * 4 * 0.10)))
    return json.dumps({
        "trigger_on_total_tokens": trigger,
        "max_history_chars": keep_chars,
        "input_budget_trigger_ratio": config.HISTORY_SUMMARY_INPUT_BUDGET_RATIO,
    })


def codegpt_model_ids() -> list[str]:
    """Models the CodeGPT Plus plan exposes, read from the extension's local DB.

    Falls back to the known plan list so Auggie always has something to pick
    even when the CodeGPT extension has not been opened on this machine yet.
    """
    # The inclusive ("economy") tier, read live from the CodeGPT catalog so the
    # list follows the plan instead of being pinned here.
    from . import codegpt

    return [entry["id"] for entry in codegpt.load_catalog_models()]

def fake_models() -> dict[str, Any]:
    """Builds comprehensive model list including all 9router combos, aliases, and catalog."""
    target_context = effective_context_limit(config.TARGET_MODEL)
    models_list = [model_list_entry(config.TARGET_MODEL, target_context)]
    model_registry: dict[str, Any] = {
        config.TARGET_MODEL: {
            **model_registry_entry(
                config.TARGET_MODEL,
                description=f"{'9router' if config.IS_9ROUTER else 'OpenAI-compatible'} model via auggie-launch",
            ),
            "context": target_context,
            "maxOutput": config.MODEL_MAX_OUTPUT_TOKENS,
        }
    }

    seen_models: set[str] = {config.TARGET_MODEL}

    # 1. Inject 9router combos from local db.json (only when 9router is the upstream)
    for combo in (config._LOCAL_9ROUTER.combos if config.IS_9ROUTER else []):
        cname = combo.get("name")
        if not cname or cname in seen_models:
            continue
        seen_models.add(cname)
        cmodels = combo.get("models") or []
        combo_context = combo_context_limit(combo)
        models_list.append(model_list_entry(cname, combo_context))
        model_registry[cname] = {
            **model_registry_entry(
                cname,
                description=f"Auto-fallback combo over {len(cmodels)} models: {', '.join(cmodels[:3])}...",
                group="9router combos",
            ),
            "context": combo_context,
            "maxOutput": config.MODEL_MAX_OUTPUT_TOKENS,
        }

    # 2. Inject 9router model aliases from local db.json (only when 9router is active)
    for alias_name, real_target in (config._LOCAL_9ROUTER.model_aliases if config.IS_9ROUTER else {}).items():
        if not alias_name or alias_name in seen_models:
            continue
        seen_models.add(alias_name)
        alias_context = effective_context_limit(alias_name)
        models_list.append(model_list_entry(alias_name, alias_context))
        model_registry[alias_name] = {
            **model_registry_entry(
                alias_name,
                description=f"Alias pointing to {real_target}",
                group="9router aliases",
            ),
            "context": alias_context,
            "maxOutput": config.MODEL_MAX_OUTPUT_TOKENS,
        }

    # 3. Inject CodeGPT Plus agent models (the agent-backed cloud has no /models)
    if config.IS_CODEGPT:
        for mid in codegpt_model_ids():
            if mid in seen_models:
                continue
            seen_models.add(mid)
            mid_context = effective_context_limit(mid)
            models_list.append(model_list_entry(mid, mid_context))
            model_registry[mid] = {
                **model_registry_entry(
                    mid,
                    description="CodeGPT Plus cloud model (routed through the agent)",
                    group="CodeGPT Plus",
                ),
                "context": mid_context,
                "maxOutput": config.MODEL_MAX_OUTPUT_TOKENS,
            }

    # 4. Inject live models from upstream /v1/models if enabled
    if config.DYNAMIC_MODELS and not config.IS_CODEGPT:
        dynamic_list = fetch_upstream_models()
        for item in dynamic_list:
            mid = item.get("id")
            if not mid or mid in seen_models:
                continue
            seen_models.add(mid)
            mid_context = effective_context_limit(mid)
            models_list.append(model_list_entry(mid, mid_context))
            model_registry[mid] = {
                **model_registry_entry(
                    mid,
                    description=f"Model from {'9router' if config.IS_9ROUTER else 'upstream'}",
                    group="upstream",
                ),
                "context": mid_context,
                "maxOutput": config.MODEL_MAX_OUTPUT_TOKENS,
            }

    all_names = ",".join(seen_models)

    return {
        "default_model": config.TARGET_MODEL,
        "models": models_list,
        "languages": [
            {"name": "TypeScript", "vscode_name": "typescript", "extensions": [".ts", ".tsx"]},
            {"name": "JavaScript", "vscode_name": "javascript", "extensions": [".js", ".jsx", ".mjs"]},
            {"name": "Python", "vscode_name": "python", "extensions": [".py"]},
            {"name": "Markdown", "vscode_name": "markdown", "extensions": [".md"]},
            {"name": "JSON", "vscode_name": "json", "extensions": [".json"]},
            {"name": "Go", "vscode_name": "go", "extensions": [".go"]},
            {"name": "Rust", "vscode_name": "rust", "extensions": [".rs"]},
            {"name": "Java", "vscode_name": "java", "extensions": [".java"]},
            {"name": "C++", "vscode_name": "cpp", "extensions": [".cpp", ".h", ".hpp", ".cc"]},
        ],
        "feature_flags": {
            "additional_chat_models": all_names,
            "agent_chat_model": config.TARGET_MODEL,
            "enable_model_registry": True,
            "model_info_registry": json.dumps(model_registry),
            # Enables Auggie's history summarization. It is gated on a non-empty
            # min version; without these flags the CLI never compacts a long
            # session and eventually overflows the upstream context window.
            "history_summary_min_version": config.HISTORY_SUMMARY_MIN_VERSION if config.HISTORY_SUMMARY_ENABLED else "",
            "history_summary_params": history_summary_params(),
            "enable_hindsight": False,
            "bypass_language_filter": True,
            "small_sync_threshold": 1048576,
            "big_sync_threshold": 10485760,
            "max_upload_size_bytes": 0 if config.INDEXING_MODE == "complete" else 52428800,
            "cli_enable_sentry": False,
            "beachhead_enable_sentry": False,
            "use_intake_service_for_file_walk": False,
            "cli_enable_worker_thread_path_filter": False,
            "enable_prompt_enhancer": True,
            "enable_command_suggestions": True,
            "enable_subagent_support": True,
        },
        "user_tier": "ENTERPRISE_TIER",
        "user": {"id": "user_auggie_launch_local", "email": "proxy@9router.local"},
        "bootstrap_settings": {"repository_allowlist_settings": {"repository_urls": [], "is_deny_list": False}},
    }


def fake_find_missing(body: Any) -> dict[str, Any]:
    if config.INDEXING_MODE == "complete":
        return {"unknown_memory_names": [], "nonindexed_blob_names": []}
    if isinstance(body, dict) and isinstance(body.get("mem_object_names"), list):
        names = [name for name in body["mem_object_names"] if isinstance(name, str)]
    else:
        names = []
    return {"unknown_memory_names": names, "nonindexed_blob_names": []}


def fake_batch_upload(body: Any) -> dict[str, Any]:
    if config.INDEXING_MODE == "complete":
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


