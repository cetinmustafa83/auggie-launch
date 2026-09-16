from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from . import config
from .config import log
from .upstream import get_active_key

# ============================================================================
# Full Injections Engine (Environment, Session, MCP Tools)
# ============================================================================

def provider_key(name: str) -> str:
    """Resolves a provider API key from the explicit env, then the 9router DB.

    `AUGGIE_LAUNCH_<NAME>_API_KEY` is checked first so MCP servers can be wired
    up without 9router, then the shared `<NAME>_API_KEY`, then the local 9router
    state that historically held these keys.
    """
    upper = name.upper().replace("-", "_")
    for key in (f"AUGGIE_LAUNCH_{upper}_API_KEY", f"{upper}_API_KEY"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return str(config._LOCAL_9ROUTER.provider_api_keys.get(name) or "").strip()


def mcp_command(bin_name: str, npx_package: str) -> dict[str, Any]:
    """Prefers an installed binary over `npx`.

    `npx -y pkg@latest` re-resolves the package on every launch, which takes
    ~25s here -- well past Auggie's 10s MCP startup window, so the server is
    reported as failed on every run. A global install starts immediately; npx
    stays as the fallback, with the cached copy preferred over the network.
    """
    home = os.path.expanduser("~")
    for candidate in (
        os.path.join(home, ".npm-global", "bin", bin_name),
        os.path.join("/usr", "local", "bin", bin_name),
        os.path.join(home, ".local", "bin", bin_name),
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return {"command": candidate, "args": []}
    return {"command": "npx", "args": ["-y", "--prefer-offline", npx_package]}


def generate_injected_mcp_config() -> str:
    """Generates an MCP configuration file for Auggie CLI with 9router's tools."""
    home = os.path.expanduser("~")
    mcp_config_path = os.path.join(tempfile.gettempdir(), "auggie_injected_mcp.json")

    mcp_servers: dict[str, Any] = {}

    # Sequential thinking MCP (installed in user's global npm)
    seq_bin = os.path.join(home, ".npm-global", "bin", "mcp-server-sequential-thinking")
    if os.path.isfile(seq_bin) or os.path.islink(seq_bin):
        mcp_servers["sequential-thinking"] = {
            "command": seq_bin,
            "args": [],
        }

    # Tavily Web Search MCP: an explicit env key wins over the 9router DB, so it
    # also works on a non-9router upstream like CodeGPT.
    tavily_key = provider_key("tavily")
    if tavily_key:
        mcp_servers["tavily"] = {
            **mcp_command("tavily-mcp", "tavily-mcp@latest"),
            "env": {"TAVILY_API_KEY": tavily_key},
        }

    # Exa MCP if configured
    exa_key = provider_key("exa")
    if exa_key:
        mcp_servers["exa"] = {
            **mcp_command("exa-mcp-server", "exa-mcp-server"),
            "env": {"EXA_API_KEY": exa_key},
        }

    try:
        with open(mcp_config_path, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": mcp_servers}, f, indent=2)
        return mcp_config_path
    except Exception as exc:
        log(f"could not write injected mcp config: {exc}")
        return ""


def build_injected_environment(proxy_url: str) -> dict[str, str]:
    """Builds complete environment variable injections for Auggie CLI."""
    env = os.environ.copy()
    active_key = get_active_key()

    # 1. Augment Core Injections
    env["AUGMENT_API_URL"] = proxy_url
    env["AUGMENT_API_TOKEN"] = config.LOCAL_TOKEN
    env["AUGMENT_SESSION_AUTH"] = json.dumps(
        {
            "accessToken": config.LOCAL_TOKEN,
            "tenantURL": proxy_url,
            "scopes": ["email", "profile", "offline_access"],
        },
        separators=(",", ":"),
    )
    env.setdefault("AUGMENT_INDEXING_MODE", config.INDEXING_MODE)
    env["AUGMENT_DISABLE_AUTO_UPDATE"] = "1"
    env["AUGMENT_MODEL"] = config.TARGET_MODEL
    env["AUGMENT_USER_AGENT"] = config.UPSTREAM_USER_AGENT

    # 2. 9router Caveman System Prompt Injections
    instructions_parts = []
    if config.ROUTER_CAVEMAN_MODE:
        instructions_parts.append(
            f"9router Caveman Mode Active ({config.ROUTER_CAVEMAN_LEVEL}): Be concise and direct. "
            "Output pure code and minimal required explanations to save tokens."
        )
    if os.environ.get("AUGMENT_INSTRUCTIONS"):
        instructions_parts.append(os.environ["AUGMENT_INSTRUCTIONS"])
    if instructions_parts:
        env["AUGMENT_INSTRUCTIONS"] = "\n\n".join(instructions_parts)

    # 3. Upstream Provider Standard Injections (for CLI tools & SDKs inside Auggie)
    env["OPENAI_BASE_URL"] = config.TARGET_BASE_URL
    env["OPENAI_API_KEY"] = active_key
    env["ANTHROPIC_BASE_URL"] = config.TARGET_BASE_URL
    env["ANTHROPIC_AUTH_TOKEN"] = active_key

    # 4. Explicit provider keys (works without 9router; also forwarded to child tools).
    for prov_name, env_names in {
        "tavily": ("TAVILY_API_KEY",),
        "exa": ("EXA_API_KEY",),
        "firecrawl": ("FIRECRAWL_API_KEY",),
    }.items():
        pkey = provider_key(prov_name)
        if pkey:
            for env_name in env_names:
                env.setdefault(env_name, pkey)

    # 5. 9router Provider API Key Injections (all active providers, not just well-known)
    for prov_name, pkey in config._LOCAL_9ROUTER.provider_api_keys.items():
        norm = prov_name.lower().replace("-", "_").replace(" ", "_")
        if norm == "tavily":
            env["TAVILY_API_KEY"] = pkey
        elif norm == "firecrawl":
            env["FIRECRAWL_API_KEY"] = pkey
        elif norm in ("jina_reader", "jina"):
            env["JINA_API_KEY"] = pkey
        elif norm == "minimax":
            env["MINIMAX_API_KEY"] = pkey
        elif norm in ("kilocode",):
            env["KILOCODE_TOKEN"] = pkey
        elif norm in ("gemini-cli", "gemini"):
            env["GEMINI_API_KEY"] = pkey
            env["GOOGLE_API_KEY"] = pkey
        elif norm == "qoder":
            env["QODER_TOKEN"] = pkey
        elif norm == "ollama":
            env["OLLAMA_API_KEY"] = pkey
        else:
            # Generic passthrough: PROVIDER_API_KEY
            env[f"{norm.upper()}_API_KEY"] = pkey

    # 5. Ensure global npm bin and local bin are on PATH
    home = os.path.expanduser("~")
    extra_paths = [
        os.path.join(home, ".npm-global", "bin"),
        os.path.join(home, ".local", "bin"),
        "/usr/local/bin",
    ]
    cur_path = env.get("PATH", "")
    for ep in extra_paths:
        if os.path.isdir(ep) and ep not in cur_path.split(":"):
            cur_path = f"{ep}:{cur_path}"
    env["PATH"] = cur_path

    return env


