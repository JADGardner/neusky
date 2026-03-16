#!/bin/bash
# Entrypoint for the NeuSky research container.
# Installs mounted code editably (fast, no deps) then runs the user command.
set -e

# Ensure HOME exists and is writable
export HOME="${HOME:-/tmp/home}"
mkdir -p "$HOME" 2>/dev/null || true

# Copy wandb credentials if available
if [ -f /tmp/.netrc ]; then
    cp -f /tmp/.netrc "$HOME/.netrc" 2>/dev/null || true
fi

# Activate conda (disable -e temporarily; nerfstudio completions have syntax issues)
set +e
eval "$(conda shell.bash hook)"
conda activate research 2>/dev/null
set -e

# Install mounted code packages editably (--no-deps: all deps are in the image)
# ns_reni first (neusky depends on it), then neusky itself
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
for pkg in "$PROJECT_ROOT/ns_reni" "$PROJECT_ROOT"; do
    if [ -f "$pkg/pyproject.toml" ] || [ -f "$pkg/setup.py" ]; then
        rm -rf "$pkg"/*.egg-info 2>/dev/null || true
        pip install -e "$pkg" --no-deps --quiet 2>/dev/null || true
    fi
done

# Run whatever command was passed
exec "$@"
