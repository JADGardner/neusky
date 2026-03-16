#!/usr/bin/env bash
# .apptainer/apptainer.sh — Unified Apptainer wrapper for NeuSky container.
#
# Commands:
#   build   Build (or rebuild) the SIF image and overlay
#   install Register local code or install pip packages into /overlay-packages
#   shell   Open an interactive shell inside the container
#   exec    Run a command inside the container
#
# Flags:
#   --ro                            Force read-only overlay (auto-enabled in SLURM)
#   --                              Separator before command (for exec/install)
#
# Examples:
#   .apptainer/apptainer.sh build
#   .apptainer/apptainer.sh install
#   .apptainer/apptainer.sh install -- tensorly fpsample
#   .apptainer/apptainer.sh shell
#   .apptainer/apptainer.sh exec -- python -c "import torch"
#   .apptainer/apptainer.sh exec -- ns-train neusky --vis wandb

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load .env if present
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# Defaults (can be overridden by .env)
CONTAINER_DIR="${CONTAINER_DIR:-${SCRIPT_DIR}}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data}"
OUTPUTS_PATH="${OUTPUTS_PATH:-${PROJECT_ROOT}/outputs}"
MODEL_STORAGE_PATH="${MODEL_STORAGE_PATH:-${PROJECT_ROOT}/model-storage}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CMD=""
FORCE_RO=false
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        build|install|shell|exec)
            CMD="$1"; shift ;;
        --ro)
            FORCE_RO=true; shift ;;
        --)
            shift; PASSTHROUGH_ARGS=("$@"); break ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 {build|install|shell|exec} [--ro] [-- cmd...]" >&2
            exit 1 ;;
    esac
done

if [[ -z "${CMD}" ]]; then
    echo "Usage: $0 {build|install|shell|exec} [--ro] [-- cmd...]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
DEF_FILE="${SCRIPT_DIR}/research.def"
SIF_FILE="${CONTAINER_DIR}/research.sif"
OVERLAY_FILE="${CONTAINER_DIR}/research_overlay.img"

# Auto-enable read-only overlay inside SLURM jobs
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    FORCE_RO=true
fi

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
do_build() {
    mkdir -p "${CONTAINER_DIR}"

    if [[ ! -f "${DEF_FILE}" ]]; then
        echo "ERROR: Definition file not found: ${DEF_FILE}" >&2
        exit 1
    fi

    # -- SIF --
    if [[ -f "${SIF_FILE}" ]]; then
        echo "=== SIF already exists, skipping build ==="
        echo "Size: $(du -h "${SIF_FILE}" | cut -f1)"
    else
        echo "=== Building SIF: research ==="
        echo "Definition: ${DEF_FILE}"
        echo "Output:     ${SIF_FILE}"
        echo "Started:    $(date)"
        echo ""
        apptainer build --fix-perms --fakeroot --notest "${SIF_FILE}" "${DEF_FILE}"
        echo ""
        echo "=== SIF build complete ==="
        echo "Size: $(du -h "${SIF_FILE}" | cut -f1)"
    fi

    # -- Overlay --
    if [[ -f "${OVERLAY_FILE}" ]]; then
        echo "=== Overlay already exists, skipping creation ==="
        echo "Size: $(du -h "${OVERLAY_FILE}" | cut -f1)"
    else
        echo ""
        echo "=== Creating 4GB ext3 overlay image ==="
        apptainer overlay create --size 4096 "${OVERLAY_FILE}"
        echo "Overlay: $(du -h "${OVERLAY_FILE}" | cut -f1)"
    fi

    # Make overlay read-only at filesystem level
    chmod 444 "${OVERLAY_FILE}"

    echo ""
    echo "=== Done ==="
    echo "Container: ${SIF_FILE} ($(du -h "${SIF_FILE}" | cut -f1))"
    echo "Overlay:   ${OVERLAY_FILE} ($(du -h "${OVERLAY_FILE}" | cut -f1), read-only)"
}

# ---------------------------------------------------------------------------
# install — install packages and register local code into /overlay-packages
# ---------------------------------------------------------------------------
do_install() {
    if [[ ! -f "${SIF_FILE}" ]]; then
        echo "ERROR: SIF not found: ${SIF_FILE}" >&2
        echo "Run: $0 build" >&2
        exit 1
    fi

    if [[ ! -f "${OVERLAY_FILE}" ]]; then
        echo "ERROR: Overlay not found: ${OVERLAY_FILE}" >&2
        echo "Run: $0 build" >&2
        exit 1
    fi

    # Make overlay writable for install
    chmod 644 "${OVERLAY_FILE}" 2>/dev/null || true

    # Bind code as :rw so pip can write .egg-info into source dirs
    local args=(--no-home --nv)
    args+=(--overlay "${OVERLAY_FILE}")
    args+=(-B "${PROJECT_ROOT}:/workspace/code")
    args+=(-B "${DATA_PATH}:/workspace/data")
    args+=(-B "${OUTPUTS_PATH}:/workspace/outputs")
    args+=(-B "${MODEL_STORAGE_PATH}:/workspace/model-storage")

    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
        # Install arbitrary pip packages: install -- pkg1 pkg2
        echo "=== Installing packages into /overlay-packages ==="
        apptainer exec "${args[@]}" "${SIF_FILE}" bash -c "
            set -euo pipefail
            mkdir -p /overlay-packages
            pip install --target /overlay-packages --no-cache-dir ${PASSTHROUGH_ARGS[*]}
        "
    else
        # Default: register local project code into /overlay-packages
        echo "=== Registering local project code in /overlay-packages ==="
        apptainer exec "${args[@]}" "${SIF_FILE}" bash -c "
            set -euo pipefail
            mkdir -p /overlay-packages

            register_if_needed() {
                local pkg_path=\"\$1\"
                local import_name=\"\$2\"
                local pkg_name=\$(basename \"\$pkg_path\")
                if ! python -c \"import \$import_name\" 2>/dev/null; then
                    echo \"  Registering \$pkg_name ...\"
                    pip install --target /overlay-packages -e \"\$pkg_path\" --no-deps --no-cache-dir --no-build-isolation
                else
                    echo \"  \$pkg_name already registered, skipping.\"
                fi
            }

            echo '[1/3] nerfstudio (mainline)'
            echo '  Registering nerfstudio ...'
            pip install --target /overlay-packages -e /opt/nerfstudio --no-deps --no-cache-dir --no-build-isolation

            echo '[2/3] ns_reni'
            register_if_needed /workspace/code/ns_reni reni

            echo '[3/3] neusky'
            register_if_needed /workspace/code neusky
        "
    fi

    # Lock overlay back to read-only
    chmod 444 "${OVERLAY_FILE}"

    echo "=== Install complete ==="
}

# ---------------------------------------------------------------------------
# Assemble bind-mount and overlay flags
# ---------------------------------------------------------------------------
build_run_args() {
    local args=(--no-home --nv)

    # Overlay: always read-only + writable-tmpfs for shell/exec
    if [[ -f "${OVERLAY_FILE}" ]]; then
        args+=(--overlay "${OVERLAY_FILE}:ro" --writable-tmpfs)
    fi

    # Bind mounts
    args+=(-B "${PROJECT_ROOT}:/workspace/code:ro")
    args+=(-B "${DATA_PATH}:/workspace/data")
    args+=(-B "${OUTPUTS_PATH}:/workspace/outputs")
    args+=(-B "${MODEL_STORAGE_PATH}:/workspace/model-storage")

    # Environment variables
    args+=(--env "PYTHONPATH=/overlay-packages")

    # Common optional mounts
    [[ -d "${HOME}/.ssh" ]]          && args+=(-B "${HOME}/.ssh:/root/.ssh:ro")
    [[ -f "${HOME}/.gitconfig" ]]    && args+=(-B "${HOME}/.gitconfig:/root/.gitconfig:ro")
    [[ -f "${HOME}/.netrc" ]]        && args+=(-B "${HOME}/.netrc:/root/.netrc:ro")
    [[ -d "${HOME}/.config/wandb" ]] && args+=(-B "${HOME}/.config/wandb:/root/.config/wandb:ro")

    echo "${args[@]}"
}

# ---------------------------------------------------------------------------
# shell / exec
# ---------------------------------------------------------------------------
do_shell() {
    if [[ ! -f "${SIF_FILE}" ]]; then
        echo "ERROR: SIF not found: ${SIF_FILE}" >&2
        echo "Run: $0 build" >&2
        exit 1
    fi

    local run_args
    read -r -a run_args <<< "$(build_run_args)"

    apptainer shell "${run_args[@]}" "${SIF_FILE}"
}

do_exec() {
    if [[ ! -f "${SIF_FILE}" ]]; then
        echo "ERROR: SIF not found: ${SIF_FILE}" >&2
        echo "Run: $0 build" >&2
        exit 1
    fi

    if [[ ${#PASSTHROUGH_ARGS[@]} -eq 0 ]]; then
        echo "ERROR: exec requires a command after --" >&2
        exit 1
    fi

    local run_args
    read -r -a run_args <<< "$(build_run_args)"

    apptainer exec "${run_args[@]}" "${SIF_FILE}" "${PASSTHROUGH_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${CMD}" in
    build)   do_build ;;
    install) do_install ;;
    shell)   do_shell ;;
    exec)    do_exec ;;
esac
