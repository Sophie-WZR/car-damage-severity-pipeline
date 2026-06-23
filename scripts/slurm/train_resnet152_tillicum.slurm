#!/bin/bash
# Train ResNet152 on Tillicum (single GPU).
# Submit with: sbatch scripts/slurm/train_resnet152_tillicum.slurm

#SBATCH --job-name=cardmg-resnet152
#SBATCH --account=macsvlarobotics
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/%u/car-damage/logs/resnet152_%j.out
#SBATCH --error=/gpfs/scrubbed/%u/car-damage/logs/resnet152_%j.err

set -euo pipefail

# ---- paths (edit ALLOC if your allocation differs) ----
ALLOC=/gpfs/projects/macsvlarobotics/car-damage
SCRUBBED=/gpfs/scrubbed/$USER/car-damage
REPO=$ALLOC/car-damage-severity-pipeline
ENV=$ALLOC/env

# Optional overrides at submit time:
# sbatch --export=ALL,EPOCHS=20,BATCH_SIZE=32 scripts/slurm/train_resnet152_tillicum.slurm
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}

# ---- offline weight cache ----
export HF_HOME=$ALLOC/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- environment ----
# Unload conda if already loaded (from login shell), then load fresh
module unload conda 2>/dev/null || true
module load conda
conda activate "$ENV"

# ---- run from repo root so manifest path resolver can find data ----
cd "$REPO"

# Safe symlink: only create if the real data directory exists; never create loops.
for split in training validation; do
    target="$REPO/data/$split"
    if [[ ! -d "$target" ]]; then
        echo "ERROR: data directory not found: $target"
        exit 1
    fi
    # Remove stale link/file before re-linking
    [[ -L "$split" || -e "$split" ]] && rm -f "$split"
    ln -s "$target" "$split"
    echo "Linked $split -> $target ($(ls $split | wc -l) subdirs)"
done

mkdir -p "$SCRUBBED/artifacts/resnet152"

nvidia-smi -L
echo "Launching ResNet152 training: epochs=$EPOCHS batch_size=$BATCH_SIZE"

python tools/train_resnet_ddp.py \
  --variant resnet152 \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv data_quality/heldout_test_manifest.csv \
  --output-dir "$SCRUBBED/artifacts/resnet152" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --wandb-project "car-damage-severity"

echo "Done -> $SCRUBBED/artifacts/resnet152"
