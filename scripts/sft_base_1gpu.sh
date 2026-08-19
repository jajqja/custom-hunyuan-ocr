#!/bin/bash
# ============================================================================
# SFT the HunyuanOCR base model on a SINGLE 80GB GPU (A100-80 / H100).
#
# Same entry point as scripts/sft_base.sh, retuned for one GPU and for a small
# domain dataset (~1k pages) instead of a cluster and ~1M packs:
#
#   NPROC_PER_NODE  8    -> 1
#   env_common.sh        -> env_single.sh   (no InfiniBand on a single box)
#   packed_max_length    20480 -> 16384     (upstream notes 20480 OOMs at 80GB)
#   GRAD_ACCUM      1    -> 4               (1 pack/step is too noisy a batch)
#   SAVE_STEPS      200  -> 50              (a whole epoch is well under 200)
#
# Everything is still env-overridable, e.g.  PACK_LEN=20480 EPOCHS=3 bash ...
# ============================================================================

set -e

# ────────────── Common env ──────────────
source scripts/env_single.sh

# ────────────── Distributed configuration ──────────────
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NODE_RANK=${NODE_RANK:-0}

# ────────────── Model & data paths ──────────────
model_name_or_path=${MODEL_PATH:?set MODEL_PATH to the HunyuanOCR base model dir}
train_data_path=${TRAIN_DATA:-./data/packed/train_16384.jsonl}
image_path="not_needed"    # packed records carry absolute image paths

# Interpreter / launcher. Set these when the deps live in a venv that is not on
# PATH:  PYTHON=/content/venv/bin/python TORCHRUN=/content/venv/bin/torchrun
PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}

# Fail here rather than after the model has started loading. A missing
# flash_attn almost always means the wrong interpreter, not a missing install.
if ! "$PYTHON" -c "import torch, flash_attn, transformers" 2>/dev/null; then
    echo "[ERROR] $PYTHON thiếu torch / flash_attn / transformers."
    echo "        Dùng venv thì truyền vào:"
    echo "            PYTHON=/content/venv/bin/python \\"
    echo "            TORCHRUN=/content/venv/bin/torchrun bash scripts/sft_base_1gpu.sh"
    echo "        hoặc đặt PATH=/content/venv/bin:\$PATH trước lệnh."
    exit 1
fi

# ────────────── Hyperparameters ──────────────
lr=${LR:-2e-5}
batch_size=${BATCH_SIZE:-1}
grad_accum_steps=${GRAD_ACCUM:-4}
num_epochs=${EPOCHS:-5}
warmup_ratio=${WARMUP:-0.03}
save_steps=${SAVE_STEPS:-50}
# save_total_limit chỉ giữ N checkpoint MỚI NHẤT. Đặt SAVE_STEPS nhỏ + limit 3 thì
# ba cái sống sót nằm sát nhau ở cuối run và không có gì để chọn giữa chúng. Muốn
# so được thì đặt SAVE_STEPS = số step một epoch và nâng limit lên bằng số epoch.
save_limit=${SAVE_LIMIT:-3}
pack_len=${PACK_LEN:-16384}

# PACKING=0 tắt gộp nhiều trang vào một chuỗi. Bắt buộc trên transformers >= 5.13:
# PackedVLDataCollator truyền cu_seqlens qua chỗ của attention_mask, thứ chỉ có
# nghĩa với bản attention đã monkeypatch — mà bản đó không còn áp dụng được.
# TRAIN_DATA khi đó phải là JSONL sinh bởi `makedata_to_hyocr.py --flat`.
packing=${PACKING:-1}
if [ "$packing" = "1" ]; then
    pack_flags="--data_flatten True --data_packing True"
else
    pack_flags="--data_flatten False --data_packing False"
fi

# Eval trong lúc train. Chỉ chạy được khi PACKING=0: eval_dataset luôn được tạo
# với is_packed=False, nên khi packing bật thì collator là PackedVLDataCollator
# và nó vỡ ngay batch eval đầu tiên.
eval_data=${EVAL_DATA:-}
eval_steps=${EVAL_STEPS:-32}
eval_batch=${EVAL_BATCH:-${BATCH_SIZE:-1}}
eval_flags="--eval_strategy no"
if [ -n "$eval_data" ]; then
    if [ "$packing" = "1" ]; then
        echo "[ERROR] EVAL_DATA cần PACKING=0."
        exit 1
    fi
    eval_flags="--eval_data_path ${eval_data} --eval_strategy steps --eval_steps ${eval_steps}"
fi

# Which parts to train. Freezing the vision tower (TUNE_VISION=False) saves
# roughly 15% of step time and is worth trying if the pages are clean scans.
tune_vision=${TUNE_VISION:-True}
tune_mlp=${TUNE_MLP:-True}
tune_llm=${TUNE_LLM:-True}

# Optional ZeRO-2. Not needed for a 1B model at 80GB; set DEEPSPEED=scripts/zero2.json
# only if you hit OOM after lowering PACK_LEN.
deepspeed_arg=""
[ -n "${DEEPSPEED}" ] && deepspeed_arg="--deepspeed ${DEEPSPEED}"

entry_file=train/train_hunyuan.py

# ────────────── Output ──────────────
run_name=${RUN_NAME:-"hyocr_sft_1gpu_lr${lr}_ep${num_epochs}_$(date +%m%d_%H%M)"}
output_dir=${OUTPUT_DIR:-"./output/${run_name}"}
TENSORBOARD_DIR="${output_dir}/tensorboard/$(date "+%Y.%m.%d-%H.%M.%S")"
mkdir -p "${TENSORBOARD_DIR}"

echo "========================================"
echo "Run name        : ${run_name}"
echo "Output dir      : ${output_dir}"
echo "Base model      : ${model_name_or_path}"
echo "Train data      : ${train_data_path}"
echo "----- Hyperparameters -----"
echo "  gpus          : ${NPROC_PER_NODE}"
echo "  lr            : ${lr}"
echo "  epochs        : ${num_epochs}"
echo "  batch x accum : ${batch_size} x ${grad_accum_steps}"
echo "  pack length   : ${pack_len}"
echo "  save          : mỗi ${save_steps} step, giữ ${save_limit} bản mới nhất"
echo "  packing       : ${packing}"
echo "  eval          : ${eval_data:-off}${eval_data:+ mỗi ${eval_steps} step}"
echo "  tune v/m/l    : ${tune_vision} / ${tune_mlp} / ${tune_llm}"
echo "  deepspeed     : ${DEEPSPEED:-off}"
echo "========================================"

# ────────────── Training args ──────────────
args="
    --model_name_or_path ${model_name_or_path} \
    --train_data_path ${train_data_path} \
    --image_folder ${image_path} \
    ${pack_flags} \
    --tune_mm_vision ${tune_vision} \
    --tune_mm_mlp ${tune_mlp} \
    --tune_mm_llm ${tune_llm} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${eval_batch} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    ${eval_flags} \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit ${save_limit} \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 5 \
    --packed_max_length ${pack_len} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --logging_dir ${TENSORBOARD_DIR} \
    --report_to tensorboard ${deepspeed_arg}"

# ────────────── Lọc flag theo transformers đang cài ──────────────
# TrainingArguments đổi tên giữa 4.x và 5.x (warmup_ratio -> warmup_steps,
# logging_dir bị bỏ). HfArgumentParser gặp flag lạ là abort ngay, nên lọc trước.
args=$("$PYTHON" tools/filter_train_args.py ${args})

# ────────────── Launch ──────────────
"${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" \
         --master_addr="${MASTER_ADDR}" \
         --master_port="${MASTER_PORT}" \
         --node_rank="${NODE_RANK}" \
         --nnodes="${NNODES}" \
         ${entry_file} ${args} 2>&1 | tee "${output_dir}/train_${NODE_RANK}.log"
