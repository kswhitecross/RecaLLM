#!/bin/bash

# Submit evaluation jobs for all datasets and context lengths.
# Usage: ./submit_all.sh <model_name_or_path> <save_dir> [--skip_long] [extra_args...]
#
# save_dir will be expanded to: <save_dir>/<dataset>/<context_length>
#
# Data is loaded from HuggingFace Hub (kswhitecross/RecaLLM-data) by default.
# Each job runs: python -m recallm.evaluation.evaluate_vllm
#
# By default, this script prints commands (dry run). To actually submit jobs,
# replace the run_command() function below with your cluster's submit command.
#
# Default args (user extra_args come after and can override):
#   --temperature 0.6
#   --top_p 0.95
#   --max_new_tokens 10240
#   --truncate_dataset 200
#   --skip_existing
#
# Short-context datasets (dapo_math, mcqa_math) only run at 4k.
# Quality only runs at 4k and 8k (identical data from 8k onward).

set -euo pipefail

# ============================================================================
# CUSTOMIZE THIS: Replace with your cluster submit command
# ============================================================================
run_command() {
    # Examples:
    #   sbatch --gres=gpu:1 --mem=80G your_wrapper.sh "$@"
    #   srun --gres=gpu:1 "$@"
    #   bash -c "$*"                # run locally (sequential)
    echo "[DRY RUN] $*"
}
# ============================================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name_or_path> <save_dir> [--skip_long] [extra_args...]"
    exit 1
fi

MODEL_NAME="$1"
BASE_SAVE_DIR="$2"
shift 2

# Parse optional args that are handled here (not forwarded)
SKIP_LONG=false
CLEANED_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--skip_long" ]]; then
        SKIP_LONG=true
        continue
    fi
    CLEANED_ARGS+=("$arg")
done
set -- "${CLEANED_ARGS[@]+"${CLEANED_ARGS[@]}"}"

# Default args (user extra_args come after and can override)
DEFAULT_ARGS=(
    --temperature 0.6
    --top_p 0.95
    --max_new_tokens 10240
    --truncate_dataset 200
    --skip_existing
)

CONTEXT_LENGTHS=(128k 96k 64k 32k 16k 8k 4k)

if $SKIP_LONG; then
    FILTERED_CONTEXTS=()
    for ctx in "${CONTEXT_LENGTHS[@]}"; do
        [[ "$ctx" == "128k" ]] && continue
        FILTERED_CONTEXTS+=("$ctx")
    done
    CONTEXT_LENGTHS=("${FILTERED_CONTEXTS[@]}")
fi

DATASETS=(
    retrieval multi_niah math_retrieval
    banking77 massive
    dapo_math mcqa_math
    quality
    majority_vote top_n_vote
    msmarco_v2
    qampari
    hotpotqa musique 2wikimultihopqa nq triviaqa
)

# Per-dataset max context: skip higher context lengths where data is identical.
declare -A MAX_CONTEXT=(
    [dapo_math]="4k"
    [mcqa_math]="4k"
    [quality]="8k"
)

# Helper: convert context name to numeric for comparison
ctx_to_num() {
    case "$1" in
        4k)   echo 4000 ;;
        8k)   echo 8000 ;;
        16k)  echo 16000 ;;
        32k)  echo 32000 ;;
        64k)  echo 64000 ;;
        96k)  echo 96000 ;;
        128k) echo 128000 ;;
        *)    echo 999999 ;;
    esac
}

SUBMITTED=0
SKIPPED=0

for context_length in "${CONTEXT_LENGTHS[@]}"; do
    ctx_num=$(ctx_to_num "$context_length")

    for dataset in "${DATASETS[@]}"; do
        # Skip if context exceeds dataset's max
        if [[ -n "${MAX_CONTEXT[$dataset]+x}" ]]; then
            max_num=$(ctx_to_num "${MAX_CONTEXT[$dataset]}")
            if (( ctx_num > max_num )); then
                continue
            fi
        fi

        run_command python -m recallm.evaluation.evaluate_vllm \
            --model "$MODEL_NAME" \
            --dataset "$dataset" \
            --context_length "$context_length" \
            --save_path "$BASE_SAVE_DIR/$dataset/$context_length" \
            "${DEFAULT_ARGS[@]}" \
            "$@"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "Generated $SUBMITTED commands ($SKIPPED skipped)"
