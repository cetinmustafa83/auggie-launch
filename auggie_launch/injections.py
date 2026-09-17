from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from . import config
from .config import log
from .transform import build_system_prompt
from .upstream import get_active_key

# ============================================================================
# Full Injections Engine (Environment, Session, MCP Tools)
# ============================================================================

def provider_key(name: str) -> str:
    """Resolves a provider API key from the environment.

    `AUGGIE_LAUNCH_<NAME>_API_KEY` is checked first, then the shared
    `<NAME>_API_KEY`.
    """
    upper = name.upper().replace("-", "_")
    for key in (f"AUGGIE_LAUNCH_{upper}_API_KEY", f"{upper}_API_KEY"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


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
    """Generates the MCP configuration file handed to the CLI."""
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

    # Tavily Web Search MCP, wired from an explicit env key.
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
    # The extra system prompt the CLI prepends to each request. Built here (not
    # only upstream-side) so the CLI's own prompt is aware of it too.
    instructions = build_system_prompt()
    if instructions:
        env["AUGMENT_INSTRUCTIONS"] = instructions

    # 3. Upstream Provider Standard Injections (for CLI tools & SDKs inside Auggie)
    env["OPENAI_BASE_URL"] = config.TARGET_BASE_URL
    env["OPENAI_API_KEY"] = active_key
    env["ANTHROPIC_BASE_URL"] = config.TARGET_BASE_URL
    env["ANTHROPIC_AUTH_TOKEN"] = active_key

    # Explicit provider keys, also forwarded to child processes.
    for prov_name, env_names in {
        "tavily": ("TAVILY_API_KEY",),
        "exa": ("EXA_API_KEY",),
        "firecrawl": ("FIRECRAWL_API_KEY",),
    }.items():
        pkey = provider_key(prov_name)
        if not pkey:
            continue
        explicit = (os.environ.get(f"AUGGIE_LAUNCH_{prov_name.upper()}_API_KEY") or "").strip()
        for env_name in env_names:
            # An explicitly prefixed key is authoritative; otherwise leave a
            # value the user already exported in place.
            if explicit or not env.get(env_name):
                env[env_name] = pkey


    return env


