#!/bin/bash
# Data-parallel (DDP) training of a single larger ResNet on ONE Tillicum node.
# One process per GPU via torchrun, gradients all-reduced each step.
#
# Submit (default resnet152 on 4 GPUs):
#   sbatch scripts/slurm/train_resnet_ddp_tillicum.slurm
# Override the variant / GPU count:
#   sbatch --gpus=8 --export=ALL,VARIANT=resnet101 scripts/slurm/train_resnet_ddp_tillicum.slurm
#
# See docs/TILLICUM_PARALLELIZATION.md. NOTE: with 1,381 training images, DDP gives only
# a marginal speedup — task parallelism (compare_resnet_array_tillicum.slurm) is usually
# the better use of multiple GPUs. Use DDP when the dataset/augmentation grows.

#SBATCH --job-name=cardmg-ddp
#SBATCH --account=macsvlarobotics
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/%u/car-damage/logs/ddp_%j.out
#SBATCH --error=/gpfs/scrubbed/%u/car-damage/logs/ddp_%j.err

set -euo pipefail

# ---- paths (edit ALLOC if your allocation differs) ----
ALLOC=/gpfs/projects/macsvlarobotics/car-damage
SCRUBBED=/gpfs/scrubbed/$USER/car-damage
REPO=$ALLOC/car-damage-severity-pipeline
ENV=$ALLOC/env

VARIANT=${VARIANT:-resnet152}          # override via --export=ALL,VARIANT=resnet101
NGPU=$(nvidia-smi -L | wc -l)          # derive from the GPUs SLURM actually granted

# ---- offline weight cache (pre-populated on the login node; see docs §weights) ----
export HF_HOME=$ALLOC/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- environment (torch ships its own CUDA libs; do NOT load the cuda module) ----
module load conda
conda activate "$ENV"

# ---- run from repo root so the manifest path-resolver finds the images ----
cd "$REPO"
ln -sfn "$SCRUBBED/data/training"   training
ln -sfn "$SCRUBBED/data/validation" validation
mkdir -p "$SCRUBBED/artifacts/$VARIANT"

nvidia-smi -L
echo "Launching DDP: variant=$VARIANT nproc_per_node=$NGPU"

# --standalone = single-node rendezvous (no master addr/port juggling needed).
torchrun --standalone --nproc_per_node="$NGPU" tools/train_resnet_ddp.py \
  --variant "$VARIANT" \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv  data_quality/heldout_test_manifest.csv \
  --output-dir "$SCRUBBED/artifacts/$VARIANT" \
  --epochs 12 \
  --batch-size 64 \
  --num-workers 4

echo "Done -> $SCRUBBED/artifacts/$VARIANT"
