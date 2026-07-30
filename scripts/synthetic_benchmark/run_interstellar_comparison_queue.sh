#!/usr/bin/env bash
set -euo pipefail

SCENE="${SCENE:-interstellar_house}"
IMAGE="${IMAGE:-neusky-research}"
NERF_CONTAINER="${NERF_CONTAINER:-nerfosr-interstellar-synthetic-train}"
NERF_RENDER_CHUNK_SIZE="${NERF_RENDER_CHUNK_SIZE:-1024}"

HOST_NEUSKY="${HOST_NEUSKY:-/home/james/GitHub/phd/code/neusky}"
HOST_NS_RENI="${HOST_NS_RENI:-/home/james/GitHub/phd/code/ns_reni}"
HOST_OUTPUTS="${HOST_OUTPUTS:-/home/james/GitHub/phd/code/neusky/outputs}"
HOST_DATA="${HOST_DATA:-/home/james/data}"
UID_GID="${UID_GID:-$(id -u):$(id -g)}"

DATASET="/workspace/data/neusky_synthetic_data/renders/${SCENE}_prepared"
NERF_CONFIG="/workspace/neusky/thirdparty/nerf-osr/configs/neusky_synthetic/${SCENE}.txt"
NERF_PRED_DIR="/workspace/outputs/synthetic_benchmark/nerf_osr/${SCENE}_gt"
GSIR_DATA="/workspace/data/neusky_synthetic_data/gs_ir_synthetic/${SCENE}"
GSIR_MODEL_PATH="/workspace/outputs/synthetic_benchmark/gs_ir_logs/${SCENE}_stage1"
GSIR_PRED_DIR="/workspace/outputs/synthetic_benchmark/gs_ir/${SCENE}_stage1"

DOCKER_BASE=(
  --user "${UID_GID}"
  --ipc=host
  --shm-size=8g
  -e HOME=/tmp/home
  -e PYTHONPATH=/workspace/neusky:/workspace/ns_reni
  -v "${HOST_NEUSKY}:/workspace/neusky"
  -v "${HOST_NS_RENI}:/workspace/ns_reni"
  -v "${HOST_OUTPUTS}:/workspace/outputs"
  -v "${HOST_DATA}:/workspace/data"
)

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

run_cpu() {
  docker run --rm "${DOCKER_BASE[@]}" -w /workspace/neusky "${IMAGE}" "$@"
}

run_gpu_neusky() {
  docker run --rm --gpus all "${DOCKER_BASE[@]}" -w /workspace/neusky "${IMAGE}" "$@"
}

run_gpu_nerfosr() {
  docker run --rm --gpus all "${DOCKER_BASE[@]}" -w /workspace/neusky/thirdparty/nerf-osr "${IMAGE}" "$@"
}

run_gpu_gsir() {
  docker run --rm --gpus all "${DOCKER_BASE[@]}" -w /workspace/neusky/thirdparty/gs-ir "${IMAGE}" "$@"
}

wait_for_nerfosr() {
  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "${NERF_CONTAINER}"; then
    log "NeRF-OSR container ${NERF_CONTAINER} not found; assuming checkpoint already exists."
    return
  fi

  local status
  if [[ "$(docker inspect -f '{{.State.Running}}' "${NERF_CONTAINER}")" == "true" ]]; then
    log "Waiting for ${NERF_CONTAINER} to finish."
    status="$(docker wait "${NERF_CONTAINER}")"
  else
    status="$(docker inspect -f '{{.State.ExitCode}}' "${NERF_CONTAINER}")"
  fi

  if [[ "${status}" != "0" ]]; then
    log "${NERF_CONTAINER} exited with status ${status}; stopping queue."
    exit 1
  fi
  log "${NERF_CONTAINER} finished cleanly."
}

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    log "Required path is missing: ${path}"
    exit 1
  fi
}

render_and_eval_nerfosr() {
  log "Rendering NeRF-OSR predictions for ${SCENE}."
  run_gpu_neusky \
    python scripts/synthetic_benchmark/render_nerfosr_predictions.py \
      --config "${NERF_CONFIG}" \
      --pred-dir "${NERF_PRED_DIR}" \
      --split test \
      --resolution-level 1 \
      -- \
      --chunk_size "${NERF_RENDER_CHUNK_SIZE}"

  log "Evaluating NeRF-OSR predictions."
  run_cpu \
    python scripts/synthetic_benchmark/evaluate.py \
      --pred-dir "${NERF_PRED_DIR}" \
      --data "${DATASET}" \
      --split test \
      --tracks nvs decomposition \
      --output "${NERF_PRED_DIR}/metrics.json" \
      --csv "${NERF_PRED_DIR}/metrics_per_frame.csv"
}

train_render_eval_gsir() {
  if [[ -e "${HOST_OUTPUTS}/synthetic_benchmark/gs_ir_logs/${SCENE}_stage1/chkpnt30000.pth" ]]; then
    log "GS-IR stage-1 checkpoint already exists; skipping training."
  else
    log "Training GS-IR stage-1 direct-render baseline for ${SCENE}."
    run_gpu_gsir \
      python train.py \
        -s "${GSIR_DATA}" \
        -m "${GSIR_MODEL_PATH}" \
        --iterations 30000 \
        --checkpoint_iterations 30000 \
        --test_iterations 7000 30000 \
        --save_iterations 30000 \
        --eval \
        --data_device cpu
  fi

  log "Rendering GS-IR stage-1 predictions for ${SCENE}."
  run_gpu_neusky \
    python scripts/synthetic_benchmark/render_gsir_predictions.py \
      --data "${GSIR_DATA}" \
      --model-path "${GSIR_MODEL_PATH}" \
      --pred-dir "${GSIR_PRED_DIR}" \
      --split test \
      --resolution 1

  log "Evaluating GS-IR stage-1 predictions."
  run_cpu \
    python scripts/synthetic_benchmark/evaluate.py \
      --pred-dir "${GSIR_PRED_DIR}" \
      --data "${DATASET}" \
      --split test \
      --tracks nvs decomposition \
      --output "${GSIR_PRED_DIR}/metrics.json" \
      --csv "${GSIR_PRED_DIR}/metrics_per_frame.csv"
}

main() {
  require_path "${HOST_DATA}/neusky_synthetic_data/renders/${SCENE}_prepared/transforms.json"
  require_path "${HOST_DATA}/neusky_synthetic_data/gs_ir_synthetic/${SCENE}/transforms_train.json"
  require_path "${HOST_NEUSKY}/thirdparty/nerf-osr/configs/neusky_synthetic/${SCENE}.txt"

  wait_for_nerfosr
  render_and_eval_nerfosr
  train_render_eval_gsir
  log "Comparison queue complete."
}

main "$@"
