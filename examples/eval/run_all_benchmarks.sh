#!/bin/bash
# =============================================================================
# Unified evaluation script for all 14 benchmarks
# Usage: bash examples/eval/run_all_benchmarks.sh
# Environment variables:
#   MODEL_PATH  - path to base model (default: Qwen/Qwen2.5-VL-3B-Instruct)
#   LORA_PATH   - path to LoRA adapter (optional)
#   RESULT_ROOT - root directory for results (default: results)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
LORA_PATH="${LORA_PATH:-}"
RESULT_ROOT="${RESULT_ROOT:-results}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="${RESULT_ROOT}/summary_${TIMESTAMP}.txt"

export MODEL_PATH LORA_PATH

mkdir -p "$RESULT_ROOT"

echo "============================================" | tee "$SUMMARY_FILE"
echo " VLM Benchmark Evaluation Suite (14 benchmarks)" | tee -a "$SUMMARY_FILE"
echo " Model: $MODEL_PATH" | tee -a "$SUMMARY_FILE"
echo " LoRA:  ${LORA_PATH:-none}" | tee -a "$SUMMARY_FILE"
echo " Time:  $(date)" | tee -a "$SUMMARY_FILE"
echo "============================================" | tee -a "$SUMMARY_FILE"

BENCHMARKS=(
    "chartqa:chartqa:run_chartqa_eval.sh"
    "cvbench:cvbench:run_cvbench_eval.sh"
    "vstar_bench:vstar_bench:run_vstar_bench_eval.sh"
    "muirbench:muirbench:run_muirbench_eval.sh"
    "blink:blink:run_blink_eval.sh"
    "seedbench:seedbench:run_seedbench_eval.sh"
    "pope:pope:run_pope_eval.sh"
    "gqa:gqa:run_gqa_eval.sh"
    "textvqa:textvqa:run_textvqa_eval.sh"
    "vizwiz_vqa:vizwiz_vqa:run_vizwiz_vqa_eval.sh"
    "dynamath:dynamath:run_dynamath_eval.sh"
    "emma:emma:run_emma_eval.sh"
    "logicvista:logicvista:run_logicvista_eval.sh"
    "flickr30k:flickr30k:run_flickr30k_eval.sh"
)

PASSED=0
FAILED=0
SKIPPED=0

for entry in "${BENCHMARKS[@]}"; do
    IFS=':' read -r bench_name bench_dir script_name <<< "$entry"
    echo ""
    echo ">>> [$bench_name] Starting..." | tee -a "$SUMMARY_FILE"

    export OUTPUT_DIR="${RESULT_ROOT}/${bench_name}"
    SCRIPT_PATH="${SCRIPT_DIR}/${bench_dir}/${script_name}"

    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "    SKIPPED: $SCRIPT_PATH not found" | tee -a "$SUMMARY_FILE"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    START_TIME=$(date +%s)
    if bash "$SCRIPT_PATH"; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "    PASSED (${ELAPSED}s)" | tee -a "$SUMMARY_FILE"
        PASSED=$((PASSED + 1))

        SCORES_FILE="${OUTPUT_DIR}/${bench_name}_scores.json"
        if [ -f "$SCORES_FILE" ]; then
            echo "    Scores: $(cat "$SCORES_FILE")" | tee -a "$SUMMARY_FILE"
        fi
    else
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "    FAILED (${ELAPSED}s)" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
    fi
done

echo "" | tee -a "$SUMMARY_FILE"
echo "============================================" | tee -a "$SUMMARY_FILE"
echo " Summary: $PASSED passed, $FAILED failed, $SKIPPED skipped" | tee -a "$SUMMARY_FILE"
echo " Results saved to: $RESULT_ROOT" | tee -a "$SUMMARY_FILE"
echo " Summary file: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "============================================" | tee -a "$SUMMARY_FILE"
