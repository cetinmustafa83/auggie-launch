#!/usr/bin/env bash
# Copyright (c) 2026 auggie-launch contributors
# Learning and research only. Any other use is at your own risk.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${AUGGIE_LAUNCH_BIN_DIR:-$HOME/.local/bin}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR="${AUGGIE_LAUNCH_CONFIG_DIR:-$XDG_CONFIG_HOME/auggie-launch}"
USER_ENV="$CONFIG_DIR/.env"
EXAMPLE="$ROOT/.env.example"
WRAPPER="$BIN_DIR/auggie-launch"

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Install auggie-launch into ~/.local/bin and optionally create a user .env.

Options:
  --bin-dir DIR     Install wrapper here (default: ~/.local/bin)
  --config-dir DIR  Config directory (default: ~/.config/auggie-launch)
  --force-env       Overwrite existing user .env from .env.example
  --no-env          Skip creating user .env
  --skip-cli        Do not auto-install the upstream CLI if missing
  --skip-9router    Do not install/verify 9router
  --skip-verify     Skip the post-install self-test
  --link            Symlink package auggie-launch instead of a small wrapper
  --pipx            Also install the package into an isolated pipx environment
  -h, --help        Show this help

After install:
  1. Edit ~/.config/auggie-launch/.env
  2. Ensure ~/.local/bin is on PATH
  3. Run: auggie-launch --print "hi"
EOF
}

FORCE_ENV=0
SKIP_CLI=0
NO_ENV=0
USE_LINK=0
SKIP_9ROUTER=0
SKIP_VERIFY=0
USE_PIPX=0
MIN_PY_MINOR=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin-dir) BIN_DIR="$2"; shift 2 ;;
    --config-dir) CONFIG_DIR="$2"; USER_ENV="$CONFIG_DIR/.env"; shift 2 ;;
    --force-env) FORCE_ENV=1; shift ;;
    --no-env) NO_ENV=1; shift ;;
    --skip-cli) SKIP_CLI=1; shift ;;
    --skip-9router) SKIP_9ROUTER=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    --pipx) USE_PIPX=1; shift ;;
    --link) USE_LINK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done



WRAPPER="$BIN_DIR/auggie-launch"
USER_ENV="$CONFIG_DIR/.env"

if [[ ! -f "$ROOT/auggie_launch.py" ]]; then
  echo "error: auggie_launch.py not found in $ROOT" >&2
  exit 1
fi

if [[ ! -f "$ROOT/auggie-launch" ]]; then
  echo "error: auggie-launch entrypoint not found in $ROOT" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required" >&2
  exit 1
fi

PY_MINOR="$(python3 -c 'import sys; print(sys.version_info[1])')"
PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info[0])')"
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -lt "$MIN_PY_MINOR" ]]; then
  echo "error: Python 3.${MIN_PY_MINOR}+ required, found $(python3 -V 2>&1)" >&2
  exit 1
fi
echo "python: $(python3 -V 2>&1) ($(command -v python3))"

if ! python3 -m py_compile "$ROOT/auggie_launch.py"; then
  echo "error: auggie_launch.py does not compile; refusing to install" >&2
  exit 1
fi

mkdir -p "$BIN_DIR" "$CONFIG_DIR"

# --- ensure required CLI is installed (auto-install when missing) ---
ensure_path_bin() {
  local dir="$1"
  [[ -n "${dir:-}" ]] || return 0
  case ":$PATH:" in
    *":$dir:"*) ;;
    *) export PATH="$dir:$PATH" ;;
  esac
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_npm_global() {
  local pkg="$1"
  if ! have_cmd npm; then
    echo "error: npm is required to install $pkg. Install Node.js/npm first:" >&2
    echo "  https://nodejs.org/  or  https://docs.npmjs.com/downloading-and-installing-node-js-and-npm" >&2
    return 1
  fi
  echo "installing CLI via npm: $pkg"
  if [[ "$(id -u)" -ne 0 ]]; then
    npm install -g "$pkg" --prefix "${HOME}/.local" || npm install -g "$pkg"
  else
    npm install -g "$pkg" || {
      echo "retry npm install with --prefix ${HOME}/.local"
      npm install -g "$pkg" --prefix "${HOME}/.local"
    }
  fi
  local npm_bin
  npm_bin="$(npm prefix -g 2>/dev/null)/bin"
  [[ -d "$npm_bin" ]] && ensure_path_bin "$npm_bin"
  ensure_path_bin "${HOME}/.local/bin"
  hash -r 2>/dev/null || true
}

run_curl_bash() {
  local url="$1"
  local label="$2"
  if ! have_cmd curl && ! have_cmd wget; then
    echo "error: curl or wget required to install $label" >&2
    return 1
  fi
  echo "installing $label via $url"
  if have_cmd curl; then
    curl -fsSL "$url" | bash
  else
    wget -qO- "$url" | bash
  fi
  ensure_path_bin "${HOME}/.local/bin"
  ensure_path_bin "${HOME}/.grok/bin"
  hash -r 2>/dev/null || true
}

ensure_required_cli() {
  if [[ "${SKIP_CLI:-0}" -eq 1 ]]; then
    echo "skip CLI auto-install (--skip-cli)"
    return 0
  fi
  ensure_path_bin "${HOME}/.local/bin"
  ensure_path_bin "${BIN_DIR:-${HOME}/.local/bin}"
  if have_cmd "auggie"; then
    echo "CLI present: auggie -> $(command -v auggie 2>/dev/null || command -v auggie)"
    return 0
  fi
  echo
  echo "missing CLI: auggie (Auggie (Augment Code CLI))"
  echo "docs: https://docs.augmentcode.com/cli/overview"
  echo "attempting automatic install..."

  run_npm_global "@augmentcode/auggie"
  if ! have_cmd auggie; then
    echo "error: 'auggie' still not on PATH after install." >&2
    return 1
  fi
  echo "ok: $(command -v auggie)"

}

chmod +x "$ROOT/auggie_launch.py" "$ROOT/auggie-launch"

if [[ "$USE_LINK" -eq 1 ]]; then
  ln -sfn "$ROOT/auggie-launch" "$WRAPPER"
  echo "linked $WRAPPER -> $ROOT/auggie-launch"
else
  cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
# Generated by auggie-launch install.sh. Do not put secrets here.
exec python3 "$ROOT/auggie_launch.py" "\$@"
EOF
  chmod +x "$WRAPPER"
  echo "installed $WRAPPER"
fi

if [[ "$NO_ENV" -eq 0 ]]; then
  if [[ -f "$USER_ENV" && "$FORCE_ENV" -eq 0 ]]; then
    echo "keep existing config: $USER_ENV"
  else
    if [[ ! -f "$EXAMPLE" ]]; then
      echo "warning: missing $EXAMPLE; writing minimal template" >&2
      cat >"$USER_ENV" <<'EOT'
AUGGIE_LAUNCH_BASE_URL=
AUGGIE_LAUNCH_MODEL=
AUGGIE_LAUNCH_API_KEY=
AUGGIE_LAUNCH_INDEXING_MODE=complete
EOT
    else
      cp -f "$EXAMPLE" "$USER_ENV"
    fi
    chmod 600 "$USER_ENV" 2>/dev/null || true
    echo "wrote $USER_ENV  (edit this file — fill in BASE_URL / MODEL / API_KEY)"
  fi
fi

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo
    echo "note: $BIN_DIR is not on PATH. Add to your shell rc, e.g.:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    ;;
esac



ensure_9router() {
  if [[ "${SKIP_9ROUTER:-0}" -eq 1 ]]; then
    echo "skip 9router setup (--skip-9router)"
    return 0
  fi
  if have_cmd 9router; then
    echo "9router present: $(command -v 9router)"
  else
    echo
    echo "missing 9router; installing latest release..."
    run_npm_global "9router@latest" || {
      echo "warning: 9router install failed; the launcher can still target any OpenAI-compatible URL" >&2
      return 0
    }
  fi
  # Seed ~/.9router/db.json from the bundled backup when there is no live DB.
  python3 "$ROOT/auggie_launch.py" --restore-9router-db >/dev/null 2>&1 && \
    echo "restored 9router DB from bundled backup" || true
}

install_pipx_package() {
  [[ "${USE_PIPX:-0}" -eq 1 ]] || return 0
  if ! have_cmd pipx; then
    echo "warning: pipx not found; skipping --pipx install" >&2
    return 0
  fi
  echo "installing package with pipx from $ROOT"
  pipx install --force "$ROOT"
}

verify_install() {
  if [[ "${SKIP_VERIFY:-0}" -eq 1 ]]; then
    echo "skip verification (--skip-verify)"
    return 0
  fi
  echo
  echo "verifying installation..."
  if ! "$WRAPPER" --help >/dev/null; then
    echo "error: $WRAPPER --help failed" >&2
    return 1
  fi
  echo "  [OK] wrapper runs: $WRAPPER"
  if AUGGIE_LAUNCH_AUTO_INSTALL_9ROUTER=false "$WRAPPER" --print-env >/dev/null 2>&1; then
    echo "  [OK] configuration resolves (--print-env)"
  else
    echo "  [WARN] configuration incomplete; edit $USER_ENV then run: auggie-launch --check"
  fi
}

ensure_required_cli
ensure_9router
install_pipx_package
verify_install

echo
echo "Done."
echo "  1) Edit config:  $USER_ENV"
echo "  2) Health check: auggie-launch --check"
echo "  3) Run:          auggie-launch --print \"hi\""
echo "  4) Proxy only:   auggie-launch --proxy-only"
