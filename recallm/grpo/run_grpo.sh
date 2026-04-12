#!/usr/bin/env bash
# Launch GRPO training with VeRL.
#
# Usage:
#   bash recallm/grpo/run_grpo.sh --config-name configs/qwen2 [overrides...]
#   bash recallm/grpo/run_grpo.sh --config-name configs/llama [overrides...]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
python -m verl.trainer.main_ppo --config-dir "${SCRIPT_DIR}" "$@"
