# #!/usr/bin/env bash
# # Train 2-D Diffusion-UNet Image policy on xArm RGB-only dataset
# #
# # Usage:
# #   bash scripts/train_xarm_baseline_2d.sh <alg_yaml> <task_yaml> <tag> <seed> <gpu>
# #
# # Example:
# #   bash scripts/train_xarm_baseline_2d.sh train_xarm_baseline_2d xarm_baseline_2d test_2d 42 0
# set -e

# # ─── User switches ──────────────────────────────────────────────────────────────
# DEBUG=False       # True → wandb offline + debug
# SAVE_CKPT=True    # always keep checkpoints
# # ────────────────────────────────────────────────────────────────────────────────

# ALG=${1:-train_xarm_baseline_2d}
# TASK=${2:-xarm_baseline_2d}
# TAG=${3:-$(date +%m%d)}
# SEED=${4:-42}
# GPU=${5:-0}

# EXP="${TASK}-${ALG}-${TAG}"
# RUN_DIR="runs/${EXP}_seed${SEED}"

# printf "\033[33m► GPU        : %s\n► experiment : %s\033[0m\n" "${GPU}" "${EXP}"
# if [ "${DEBUG}" = True ]; then
#   WANDB_MODE=offline
#   echo -e "\033[33mDEBUG mode (offline W&B)\033[0m"
# else
#   WANDB_MODE=online
#   echo -e "\033[33mTRAIN mode (W&B online)\033[0m"
# fi

# # ─── Locate and cd to code root ────────────────────────────────────────────────
# SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
# # now go up one (from scripts/) and into the real project
# CODE_ROOT="$( realpath "${SCRIPT_DIR}/../3D-Diffusion-Policy" )"
# cd "${CODE_ROOT}"

# export HYDRA_FULL_ERROR=1
# export CUDA_VISIBLE_DEVICES="${GPU}"

# # ─── Launch training ───────────────────────────────────────────────────────────
# accelerate launch --num_processes 1 train_2d.py \
#   --config-name "${ALG}.yaml" \
#   task="${TASK}" \
#   hydra.run.dir="${RUN_DIR}" \
#   training.debug="${DEBUG}" \
#   training.seed="${SEED}" \
#   training.device="cuda:${GPU}" \
#   exp_name="${EXP}" \
#   logging.mode="${WANDB_MODE}"

# # python train_2d.py \
# #   --config-name "${ALG}.yaml" \
# #   task="${TASK}" \
# #   hydra.run.dir="${RUN_DIR}" \
# #   training.debug="${DEBUG}" \
# #   training.seed="${SEED}" \
# #   training.device="cuda:${GPU}" \
# #   exp_name="${EXP}" \
# #   logging.mode="${WANDB_MODE}"

#!/usr/bin/env bash
# Train 2-D Diffusion-UNet Image policy on xArm RGB-only dataset
#
# Usage:
#   bash scripts/train_xarm_baseline_2d.sh <alg_yaml> <task_yaml> <tag> <seed> <gpu>
#
# Example:
#   bash scripts/train_xarm_baseline_2d.sh train_xarm_baseline_2d xarm_baseline_2d test_2d 42 0
#   bash scripts/train_xarm_baseline_2d.sh train_xarm_baseline_2d xarm_baseline_2d test_2d 42 0,1,2,3
set -e

# ─── User switches ──────────────────────────────────────────────────────────────
DEBUG=False       # True → wandb offline + debug
SAVE_CKPT=True    # always keep checkpoints
# ────────────────────────────────────────────────────────────────────────────────

ALG=${1:-train_xarm_baseline_2d}
TASK=${2:-xarm_baseline_2d}
TAG=${3:-$(date +%m%d)}
SEED=${4:-42}
GPU=${5:-0}

EXP="${TASK}-${ALG}-${TAG}"
RUN_DIR="runs/${EXP}_seed${SEED}"

# Count number of GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPU"
NUM_GPUS=${#GPU_ARRAY[@]}

printf "\033[33m► GPU(s)     : %s (count: %d)\n► experiment : %s\033[0m\n" "${GPU}" "${NUM_GPUS}" "${EXP}"
if [ "${DEBUG}" = True ]; then
  WANDB_MODE=offline
  echo -e "\033[33mDEBUG mode (offline W&B)\033[0m"
else
  WANDB_MODE=online
  echo -e "\033[33mTRAIN mode (W&B online)\033[0m"
fi

# ─── Locate and cd to code root ────────────────────────────────────────────────
SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
# now go up one (from scripts/) and into the real project
CODE_ROOT="$( realpath "${SCRIPT_DIR}/../3D-Diffusion-Policy" )"
cd "${CODE_ROOT}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU}"

# ─── Launch training ───────────────────────────────────────────────────────────
if [ ${NUM_GPUS} -gt 1 ]; then
  echo "Launching multi-GPU training with ${NUM_GPUS} GPUs"
  accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port 29502 train_2d.py \
    --config-name "${ALG}.yaml" \
    task="${TASK}" \
    hydra.run.dir="${RUN_DIR}" \
    training.debug="${DEBUG}" \
    training.seed="${SEED}" \
    exp_name="${EXP}" \
    logging.mode="${WANDB_MODE}"
else
  echo "Launching single-GPU training"
  python train_2d.py \
    --config-name "${ALG}.yaml" \
    task="${TASK}" \
    hydra.run.dir="${RUN_DIR}" \
    training.debug="${DEBUG}" \
    training.seed="${SEED}" \
    training.device="cuda:0" \
    exp_name="${EXP}" \
    logging.mode="${WANDB_MODE}"
fi