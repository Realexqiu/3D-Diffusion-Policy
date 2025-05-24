#!/usr/bin/env bash
# Train Simple-DP3 on the xArm ufactory-gripper dataset
# Usage:
#   bash scripts/train_xarm_baseline.sh <alg_cfg> <task_cfg> <tag> <seed> <gpu>
# example: bash scripts/train_xarm_baseline.sh simple_dp3 xarm_ufactory_gripper 0523 0 0
set -e

# ─── User switches ─────────────────────────────────────────────────────────
DEBUG=False        # True => wandb offline, quick debug run
save_ckpt=True     # True => keep checkpoints
# ───────────────────────────────────────────────────────────────────────────

# Command-line args
alg_cfg=${1:-simple_dp3}
task_cfg=${2:-xarm_ufactory_gripper}
run_tag=${3:-$(date +%m%d)}
seed=${4:-0}
gpu_id=${5:-0}

exp_name="${task_cfg}-${alg_cfg}-${run_tag}"
run_dir="data/outputs/${exp_name}_seed${seed}"

printf "\033[33m► GPU        : %s\n► experiment : %s\033[0m\n" "$gpu_id" "$exp_name"
if [ "$DEBUG" = True ]; then
  wandb_mode=offline
  echo -e "\033[33mDEBUG mode (offline)\033[0m"
else
  wandb_mode=online
  echo -e "\033[33mTRAIN mode (online)\033[0m"
fi

# ─── Figure out where train.py actually is ─────────────────────────────────
SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CODE_ROOT="$( realpath "$SCRIPT_DIR/../3D-Diffusion-Policy" )"

if [ ! -f "$CODE_ROOT/train.py" ]; then
  echo "ERROR: Cannot find train.py under $CODE_ROOT"
  exit 1
fi

cd "$CODE_ROOT"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="$gpu_id"

# ─── Finally launch training! ───────────────────────────────────────────────
python train.py \
  --config-name="${alg_cfg}.yaml" \
  task="${task_cfg}" \
  hydra.run.dir="${run_dir}" \
  training.debug="${DEBUG}" \
  training.seed="${seed}" \
  training.device="cuda:${gpu_id}" \
  exp_name="${exp_name}" \
  logging.mode="${wandb_mode}" \
  checkpoint.save_ckpt="${save_ckpt}"
