#!/bin/bash
# DynaMath evaluation script
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
LORA_PATH="${LORA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dynamath}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"

LORA_ARG=""
if [ -n "$LORA_PATH" ]; then LORA_ARG="--lora_path $LORA_PATH"; fi

python "$SCRIPT_DIR/eval_dynamath.py" \
    --model_path "$MODEL_PATH" $LORA_ARG \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --resume

echo "DynaMath evaluation complete. Results in $OUTPUT_DIR"
