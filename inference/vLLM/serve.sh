#!/bin/bash
# ============================================================================
# HunyuanOCR-1.5 vLLM single-GPU serving (autoregressive, OpenAI-compatible).
#
# Prerequisite: the unified inference environment is installed and activated,
# and the model weights are downloaded. See docs/inference/inference.md.
#
# Usage:
#   MODEL_PATH=/path/to/HunyuanOCR GPU=0 PORT=8000 bash serve.sh
#
# Environment variables (all overridable):
#   MODEL_PATH    model weights dir (required)
#   GPU           GPU index                     default 0
#   PORT          service port                  default 8000
#   GPU_MEM_UTIL  GPU memory fraction           default 0.9
#   MAX_MODEL_LEN context window length         default 131072
#   SERVED_NAME   served model name             default tencent/HunyuanOCR
# ============================================================================
set -e

MODEL_PATH=${MODEL_PATH:?"Please set MODEL_PATH=model weights dir"}
GPU=${GPU:-0}
PORT=${PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
SERVED_NAME=${SERVED_NAME:-tencent/HunyuanOCR}
LOG=${LOG:-vllm_${PORT}.log}
# Cờ tuỳ ý cho vllm serve, vd EXTRA_ARGS="--enforce-eager" khi torch.compile vỡ.
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "========================================"
echo "  HunyuanOCR-1.5 vLLM serve"
echo "  model         : ${MODEL_PATH}"
echo "  served-as     : ${SERVED_NAME}"
echo "  gpu / port    : ${GPU} / ${PORT}"
echo "  gpu_mem_util  : ${GPU_MEM_UTIL}"
echo "  max_model_len : ${MAX_MODEL_LEN}"
echo "  log           : ${LOG}"
echo "  extra args    : ${EXTRA_ARGS:-—}"
echo "========================================"

# Local weights: disable network lookups to speed up startup.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=${GPU} nohup vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    -tp 1 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --trust-remote-code \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_MODEL_LEN}" \
    ${EXTRA_ARGS} \
    > "${LOG}" 2>&1 &

echo "[started] pid=$!  log=${LOG}"
echo "Readiness check: curl -sf http://127.0.0.1:${PORT}/v1/models || tail -f ${LOG}"
