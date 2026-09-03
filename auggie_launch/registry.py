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
        "completion_timeout_ms": 600000,
    }


def fake_models() -> dict[str, Any]:
    """Builds comprehensive model list including all 9router combos, aliases, and catalog."""
    target_context = effective_context_limit(config.TARGET_MODEL)
    models_list = [model_list_entry(config.TARGET_MODEL, target_context)]
    model_registry: dict[str, Any] = {
        config.TARGET_MODEL: {
            "humanName": config.TARGET_MODEL,
            "description": f"{'9router' if config.IS_9ROUTER else 'OpenAI-compatible'} model via auggie-launch",
            "encoding": "o200k_base",
            "context": target_context,
            "maxOutput": config.MODEL_MAX_OUTPUT_TOKENS,
        }
    }

    seen_models: set[str] = {config.TARGET_MODEL}

    # 1. Inject 9router combos from local db.json
    for combo in config._LOCAL_9ROUTER.combos:
        cname = combo.get("name")
        if not cname or cname in seen_models:
            continue
        seen_models.add(cname)
        cmodels = combo.get("models") or []
        combo_context = combo_context_limit(combo)
        models_list.append(model_list_entry(cname, combo_context))
        model_registry[cname] = {
            "humanName": f"9router: {cname} (Combo)",
            "description": f"Auto-fallback combo over {len(cmodels)} models: {', '.join(cmodels[:3])}...",
            "encoding": "o200k_base",
            "context": combo_context,
            "maxOutput": config.MODEL_MAX_OUTPUT_TOKENS,
        }

    # 2. Inject 9router model aliases from local db.json
    for alias_name, real_target in config._LOCAL_9ROUTER.model_aliases.items():
        if not alias_name or alias_name in seen_models:
            continue
        seen_models.add(alias_name)
        alias_context = effective_context_limit(alias_name)
        models_list.append(model_list_entry(alias_name, alias_context))
        model_registry[alias_name] = {
            "humanName": f"9router: {alias_name}",
            "description": f"Alias pointing to {real_target}",
            "encoding": "o200k_base",
            "context": alias_context,
            "maxOutput": config.MODEL_MAX_OUTPUT_TOKENS,
        }

    # 3. Inject live models from upstream /v1/models if enabled
    if config.DYNAMIC_MODELS:
        dynamic_list = fetch_upstream_models()
        for item in dynamic_list:
            mid = item.get("id")
            if not mid or mid in seen_models:
                continue
            seen_models.add(mid)
            mid_context = effective_context_limit(mid)
            models_list.append(model_list_entry(mid, mid_context))
            model_registry[mid] = {
                "humanName": mid,
                "description": f"Model from {'9router' if config.IS_9ROUTER else 'upstream'}",
                "encoding": "o200k_base",
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


