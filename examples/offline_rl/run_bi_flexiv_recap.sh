#!/usr/bin/env bash

# Run the local, offline Bi-Flexiv RECAP pipeline.
#
# Required environment variables:
#   FLEXIV_DATA  LeRobot dataset root (contains data/, meta/, videos/)
#   SIGLIP_PATH  local SigLIP2 checkpoint
#   GEMMA_PATH   local Gemma3 checkpoint/tokenizer
#   PI05_PATH    local PyTorch pi0.5 checkpoint
#
# Usage:
#   bash examples/offline_rl/run_bi_flexiv_recap.sh returns
#   bash examples/offline_rl/run_bi_flexiv_recap.sh value
#   VALUE_CKPT=/path/to/value/checkpoint bash ... advantages
#   bash examples/offline_rl/run_bi_flexiv_recap.sh policy

set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export OFFLINE_RL_CONFIG="${REPO_PATH}/examples/offline_rl/config"

: "${FLEXIV_DATA:?Set FLEXIV_DATA to the LeRobot dataset root}"
: "${SIGLIP_PATH:?Set SIGLIP_PATH to the local SigLIP2 checkpoint}"
: "${GEMMA_PATH:?Set GEMMA_PATH to the local Gemma3 checkpoint}"
: "${PI05_PATH:?Set PI05_PATH to the local pi0.5 PyTorch checkpoint}"

stage="${1:-}"
case "$stage" in
  returns)
    bash "${REPO_PATH}/examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh" \
      recap_bi_flexiv_compute_returns \
      "data.train_data_paths.0.dataset_path=${FLEXIV_DATA}"
    ;;
  value)
    bash "${REPO_PATH}/examples/offline_rl/advantage_labeling/recap/run_value_sft.sh" \
      recap_bi_flexiv_value_model_sft \
      "data.train_data_paths.0.dataset_path=${FLEXIV_DATA}" \
      "actor.model.siglip_path=${SIGLIP_PATH}" \
      "actor.model.gemma3_path=${GEMMA_PATH}" \
      "actor.model.tokenizer_path=${GEMMA_PATH}"
    ;;
  advantages)
    : "${VALUE_CKPT:?Set VALUE_CKPT to the trained value-model actor checkpoint}"
    bash "${REPO_PATH}/examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh" \
      recap_bi_flexiv_compute_advantages --nproc "${NPROC:-1}" \
      "data.train_data_paths.0.dataset_path=${FLEXIV_DATA}" \
      "advantage.value_checkpoint=${VALUE_CKPT}" \
      "advantage.model.siglip_path=${SIGLIP_PATH}" \
      "advantage.model.gemma3_path=${GEMMA_PATH}" \
      "advantage.model.tokenizer_path=${GEMMA_PATH}"
    ;;
  policy)
    bash "${REPO_PATH}/examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh" \
      cfg_rl_bi_flexiv \
      "data.train_data_paths.0.dataset_path=${FLEXIV_DATA}" \
      "actor.model.model_path=${PI05_PATH}"
    ;;
  *)
    echo "Usage: $0 {returns|value|advantages|policy}" >&2
    exit 2
    ;;
esac

