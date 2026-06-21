#!/bin/bash
# Task-parallel comparison: train EACH ResNet variant on its OWN GPU, simultaneously.
# This is a SLURM job array — one independent 1-GPU job per array index. Wall-clock for
# the whole sweep collapses to the single slowest variant instead of the sum of all.
#
# Submit all five variants (indices 0..4):
#   sbatch scripts/slurm/compare_resnet_array_tillicum.slurm
# Just the three larger ones:
#   sbatch --array=2-4 scripts/slurm/compare_resnet_array_tillicum.slurm
#
# After the array finishes, aggregate the per-variant JSONs into one report:
#   python tools/analyze_resnet_comparison.py \
#     --results-dir /gpfs/scrubbed/$USER/car-damage/artifacts/compare \
#     --output-report /gpfs/scrubbed/$USER/car-damage/artifacts/compare/analysis_report.md

#SBATCH --job-name=cardmg-cmp
#SBATCH --account=macsvlarobotics
#SBATCH --qos=normal
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/%u/car-damage/logs/cmp_%A_%a.out
#SBATCH --error=/gpfs/scrubbed/%u/car-damage/logs/cmp_%A_%a.err

set -euo pipefail

# ---- paths (edit ALLOC if your allocation differs) ----
ALLOC=/gpfs/projects/macsvlarobotics/car-damage
SCRUBBED=/gpfs/scrubbed/$USER/car-damage
REPO=$ALLOC/car-damage-severity-pipeline
ENV=$ALLOC/env

# Index -> variant. Keep in sync with --array above (0-based).
VARIANTS=(resnet18 resnet34 resnet50 resnet101 resnet152)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

export HF_HOME=$ALLOC/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load conda
conda activate "$ENV"

cd "$REPO"
ln -sfn "$SCRUBBED/data/training"   training
ln -sfn "$SCRUBBED/data/validation" validation
mkdir -p "$SCRUBBED/artifacts/compare"

echo "Array task $SLURM_ARRAY_TASK_ID -> training $VARIANT"

# Each task writes <variant>_results.json (distinct filename), so tasks don't collide.
# (comparison_summary.json is per-task and not meaningful here — use analyze_* to aggregate.)
python tools/compare_resnet_variants.py \
  --train-csv data_quality/clean_train_manifest.csv \
  --test-csv  data_quality/heldout_test_manifest.csv \
  --output-dir "$SCRUBBED/artifacts/compare" \
  --variants "$VARIANT" \
  --epochs 12 \
  --batch-size 64

echo "Done $VARIANT -> $SCRUBBED/artifacts/compare/${VARIANT}_results.json"
