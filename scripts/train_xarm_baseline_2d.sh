#!/usr/bin/env bash
# Train 2-D Diffusion-UNet Image policy on xArm RGB-only dataset
#
# Usage:
#   bash scripts/train_xarm_baseline_2d.sh <alg_yaml> <task_yaml> <tag> <seed> <gpu>
#
# Example (recommended):
#   bash scripts/train_xarm_baseline_2d.sh train_xarm_baseline_2d xarm_ufactory_gripper_2d 0523 42 0
set -e

# --- switches you might touch ------------------------------------
DEBUG=False        # True → wandb offline + quick run
SAVE_CKPT=True     # always keep checkpoints for real data
# -----------------------------------------------------------------

ALG=${1:-train_xarm_baseline_2d}
TASK=${2:-xarm_ufactory_gripper_2d}
TAG=${3:-$(date +%m%d)}
SEED=${4:-42}
GPU=${5:-0}

EXP="${TASK}-${ALG}-${TAG}"
RUN_DIR="data/outputs/${EXP}_seed${SEED}"

printf "\033[33m► GPU        : %s\n► experiment : %s\033[0m\n" "${GPU}" "${EXP}"
if [ "${DEBUG}" = True ]; then
    WANDB_MODE=offline
    echo -e "\033[33mDEBUG mode (offline W&B)\033[0m"
else
    WANDB_MODE=online
    echo -e "\033[33mTRAIN mode (W&B online)\033[0m"
fi

# ─── Figure out where train.py actually is ─────────────────────────────────
SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CODE_ROOT="$( realpath "$SCRIPT_DIR/../3D-Diffusion-Policy" )"

if [ ! -f "$CODE_ROOT/train_2d.py" ]; then
  echo "ERROR: Cannot find train_2d.py under $CODE_ROOT"
  exit 1
fi

cd "$CODE_ROOT"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="$GPU"

python train_2d.py \
    --config-name="${ALG}.yaml" \
    task="${TASK}" \
    hydra.run.dir="${RUN_DIR}" \
    training.debug="${DEBUG}" \
    training.seed="${SEED}" \
    training.device="cuda:${GPU}" \
    exp_name="${EXP}" \
    logging.mode="${WANDB_MODE}"
